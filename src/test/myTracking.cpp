#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <sys/stat.h>
#include <iomanip>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ceres/ceres.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "sophus/sim3.hpp"
#include "sophus/se3.hpp"

#include "pose_local_parameterization.h"

using namespace std;
using namespace cv;

double setting_huberTH = 9.0;

void projectRefToNew(Sophus::SE3 refToNew, Eigen::Matrix3d K, Eigen::Matrix3d Ki,
                     Eigen::Vector2d pt, double idepth,
                     Eigen::Vector2d& new_pt, Eigen::Vector2d& new_Kpt, double& new_idepth)
{
    //R*K'
    Eigen::Matrix3d RKi = (refToNew.rotationMatrix() * Ki);
     //t
    Eigen::Vector3d t = refToNew.translation();

    Eigen::Vector3d Pt = RKi * Eigen::Vector3d(pt[0], pt[1], 1) + t * idepth;
    new_pt[0] = Pt[0] / Pt[2];
    new_pt[1] = Pt[1] / Pt[2];
    new_Kpt[0] = K(0, 0) * new_pt[0] + K(0, 2);
    new_Kpt[1] = K(1, 1) * new_pt[1] + K(1, 2);
    new_idepth = idepth / Pt[2];
}

EIGEN_ALWAYS_INLINE Eigen::Vector3d getInterpolatedElement33(const Eigen::Vector3d* const mat, const double x, const double y, const int width)
{
    int ix = (int)x;
    int iy = (int)y;
    double dx = x - ix;
    double dy = y - iy;
    double dxdy = dx * dy;
    const Eigen::Vector3d* bp = mat + ix + iy * width;

    return dxdy * *(const Eigen::Vector3d*)(bp + 1 + width)
           + (dy - dxdy) * *(const Eigen::Vector3d*)(bp + width)
           + (dx - dxdy) * *(const Eigen::Vector3d*)(bp + 1)
           + (1 - dx - dy + dxdy) * *(const Eigen::Vector3d*)(bp);
}

class AffLight
{
public:
    AffLight(double a_, double b_) : a(a_), b(b_) {};
    AffLight() : a(0), b(0) {};

    // Affine Parameters:
    double a, b; // I_frame = exp(a)*I_global + b. // I_global = exp(-a)*(I_frame - b).

    //用来计算两帧间的曝光差,如果没有曝光时间，则曝光时间都是1
    static Eigen::Vector2d fromToVecExposure(double exposureF, double exposureT, AffLight g2F, AffLight g2T)
    {
        if (exposureF == 0 || exposureT == 0)
        {
            exposureT = exposureF = 1;
            //printf("got exposure value of 0! please choose the correct model.\n");
            //assert(setting_brightnessTransferFunc < 2);
        }
        //refToFh[0]=a＝e^(aj-ai)*tj*ti^(-1),两帧间的光度曝光变化
        double a = exp(g2T.a - g2F.a) * exposureT / exposureF;
        //refToFh[1]=b = 当前帧的b - refToFh[0]*当前帧的b
        double b = g2T.b - a * g2F.b;
        return Eigen::Vector2d(a, b);
    }

    Eigen::Vector2d vec()
    {
        return Eigen::Vector2d(a, b);
    }
};
int numTermsInE = 0;
int numTermsInE1 = 0;
/**
 * @brief The AlignmentFactor class
 *
 * 直接法误差因子，这里使用SizedCostFunstion
 * 残差为1维，优化变量为6,2维，共2个，i帧相对与参考帧位姿，a和b
 */
class AlignmentFactor : public ceres::SizedCostFunction<1, 6, 2>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    AlignmentFactor() = delete ;
    AlignmentFactor(Eigen::Vector3d* _dINewl,
                    const Eigen::Vector2d& _pt, const double& _idepth, const double& _refColor, const AffLight& _ref_aff_g2l,
                    Eigen::Matrix3d _K, Eigen::Matrix3d _Ki, int w, int h, double _cutoffTH)
        : dINewl(_dINewl), pt(_pt), idepth(_idepth), refColor(_refColor), K(_K), Ki(_Ki), width(w), height(h),
          ref_aff_g2l(_ref_aff_g2l), cutoffTH(_cutoffTH)
    {
    }
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Matrix<double,6,1> pose;
        for ( size_t i=0; i<6; i++ )
            pose[i] = parameters[0][i];

        Sophus::SE3 init_T=Sophus::SE3 (
                    Sophus::SO3::exp( pose.tail<3>() ),
                    pose.head<3>()
                );

        AffLight aff_g2l(parameters[1][0], parameters[1][1]);

        Eigen::Vector2d new_pt;
        Eigen::Vector2d new_kpt;
        double new_idepth;

        projectRefToNew(init_T, K, Ki, pt, idepth, new_pt, new_kpt, new_idepth);

        Eigen::Vector2d affLL = AffLight::fromToVecExposure(1.0, 1.0, ref_aff_g2l, aff_g2l);
        Eigen::Vector3d hitColor(0, 0, 0);

        bool out = false ;
        if (!(new_kpt[0] > 2 && new_kpt[1] > 2 && new_kpt[0] < width - 3 && new_kpt[1] < height - 3 && new_idepth > 0))
        {
            residuals[0] = 0;
            out = true;
        }
        else
        {
            hitColor = getInterpolatedElement33(dINewl, new_kpt[0], new_kpt[1], width);
            double res = (double)(hitColor[0] - (double)(affLL[0] * refColor + affLL[1]));
            if (fabs(res) > cutoffTH)
            {
                residuals[0] = 0;
                out = true;
            }
            else
            {
                residuals[0] = res;
                //cout<<residuals[0]<<endl;
            }
        }

        if (jacobians)
        {
            if (jacobians[0])
            {
                double dx = hitColor[1] * K(0, 0);
                double dy = hitColor[2] * K(1, 1);

                Eigen::Map<Eigen::Matrix<double, 1, 6, Eigen::RowMajor> > jacobian_pose(jacobians[0]);
                jacobian_pose.setZero() ;
//                Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;
//                Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

                if ( out == false )
                {
                    jacobian_pose(0, 0) = (double)(new_idepth * dx);
                    jacobian_pose(0, 1) = (double)(new_idepth * dy);
                    jacobian_pose(0, 2) = (double)(-new_idepth * (dx * new_pt[0] + dy * new_pt[1]));
                    jacobian_pose(0, 3) = (double)(-(new_pt[0] * new_pt[1] * dx + (1 + new_pt[1] * new_pt[1]) * dy));
                    jacobian_pose(0, 4) = (double)((1 + new_pt[0] * new_pt[0]) * dx + new_pt[0] * new_pt[1] * dy);
                    jacobian_pose(0, 5) = (double)(new_pt[0] * dy - new_pt[1] * dx);
                    jacobian_pose=-jacobian_pose;

//                    jacobian_uv_ksai(0,0)=new_idepth*K(0,0);
//                    jacobian_uv_ksai(0,1)=0;
//                    jacobian_uv_ksai(0,2)=-new_idepth*new_pt[0]*K(0,0);
//                    jacobian_uv_ksai(0,3)=-new_pt[0]*new_pt[1]*K(0,0);
//                    jacobian_uv_ksai(0,4)=(1+new_pt[0]*new_pt[0])*K(0,0);
//                    jacobian_uv_ksai(0,5)=-new_pt[1]*K(0,0);

//                    jacobian_uv_ksai(1,0)=0;
//                    jacobian_uv_ksai(1,1)=new_idepth*K(1,1);
//                    jacobian_uv_ksai(1,2)=-new_idepth*new_pt[1]*K(1,1);
//                    jacobian_uv_ksai(1,3)=-(1+new_pt[1]*new_pt[1])*K(1,1);
//                    jacobian_uv_ksai(1,4)=new_pt[0]*new_pt[1]*K(1,1);
//                    jacobian_uv_ksai(1,5)=new_pt[0]*K(1,1);

//                    jacobian_pixel_uv(0,0)=hitColor[1];
//                    jacobian_pixel_uv(0,1)=hitColor[2];

//                    jacobian_pose=-jacobian_pixel_uv*jacobian_uv_ksai;
                }
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 1, 2, Eigen::RowMajor>> jacobian_ab(jacobians[1]);
                jacobian_ab.setZero();
                if ( out == false )
                {
                    jacobian_ab(0, 0) = affLL[0] * (ref_aff_g2l.b - refColor);
                    jacobian_ab(0, 1) = 1;
                    jacobian_ab=-jacobian_ab;
                }
            }
        }

        return true;
    }

    AffLight ref_aff_g2l;
    Eigen::Vector3d* dINewl;
    Eigen::Vector2d pt;
    double idepth;
    double refColor;
    Eigen::Matrix3d K, Ki;
    int width;
    int height;
    double cutoffTH;
};

#define PYR_LEVELS 6

double* pc_u[PYR_LEVELS];
double* pc_v[PYR_LEVELS];
//参考帧的点逆深度，
double* pc_idepth[PYR_LEVELS];
//参考帧的点灰度值
double* pc_color[PYR_LEVELS];
//参考帧的点个数
int pc_n[PYR_LEVELS];

int pyrLevelsUsed = 0;

Eigen::Vector3d* dINewl[PYR_LEVELS];

Eigen::Matrix4d init_T, result_T;
AffLight init_ab, result_ab,ref_ab;
Eigen::Matrix<double, 5, 1> res;
Eigen::Vector3d lastFlowIndicators;

int ww = 424;
int hh = 320;

Eigen::Matrix3d K[PYR_LEVELS], Ki[PYR_LEVELS];

int w[PYR_LEVELS], h[PYR_LEVELS];

void SplitString(const string& s, vector<string>& v, const string& c)
{
    string::size_type pos1, pos2;
    pos2 = s.find(c);
    pos1 = 0;
    while (string::npos != pos2)
    {
        v.push_back(s.substr(pos1, pos2 - pos1));

        pos1 = pos2 + c.size();
        pos2 = s.find(c, pos1);
    }
    if (pos1 != s.length())
        v.push_back(s.substr(pos1));
}

double stringTofloat(string s)
{
    stringstream ss;
    double num;
    ss << s;
    ss >> num;
    return num;
}

double stringTodouble(string s)
{
    stringstream ss;
    double num;
    ss << s;
    ss >> num;
    return num;
}

void param2eigen(double* para_Pose, double* para_AB,Eigen::Matrix4d& T, AffLight& ab)
{
    Eigen::Matrix<double,6,1> se3;
    se3<<para_Pose[0],para_Pose[1],para_Pose[2],para_Pose[3],para_Pose[4],para_Pose[5];
    Sophus::SE3 SE3_T = Sophus::SE3::exp(se3);
    T=SE3_T.matrix();
    ab.a=para_AB[0];
    ab.b=para_AB[1];
}

void eigen2para(Eigen::Matrix4d T, AffLight ab, double* para_Pose, double* para_AB)
{
    Sophus::SE3 SE3_T(T);

    Eigen::Matrix<double,6,1> pose_curr;
    pose_curr.head<3>() = SE3_T.translation();
    pose_curr.tail<3>() = SE3_T.so3().log();
    //Eigen::Matrix<double,6,1> se3=SE3_T.log();

    para_Pose[0]=pose_curr[0];
    para_Pose[1]=pose_curr[1];
    para_Pose[2]=pose_curr[2];
    para_Pose[3]=pose_curr[3];
    para_Pose[4]=pose_curr[4];
    para_Pose[5]=pose_curr[5];

//    Eigen::Vector3d Ps = T.block<3, 1>(0, 3); //.cast<double>();
//    Eigen::Matrix3d Rs = T.block<3, 3>(0, 0); //.cast<double>();
//    para_Pose[0] = Ps.x();
//    para_Pose[1] = Ps.y();
//    para_Pose[2] = Ps.z();
//    Eigen::Quaterniond q{Rs};
//    para_Pose[3] = q.x();
//    para_Pose[4] = q.y();
//    para_Pose[5] = q.z();
//    para_Pose[6] = q.w();

    para_AB[0] = ab.a;
    para_AB[1] = ab.b;
}

void readData(string path, int id)
{
    std::ifstream ifsK(path + "/K.txt");
    std::ifstream ifsKi(path + "/Ki.txt");
    std::stringstream ss;
    ss << std::setw(6) << std::setfill('0') << id;
    path = path + "/" + ss.str();
    std::ifstream ifsDepth(path + "/idepth.txt");
    std::ifstream ifsU(path + "/u.txt");
    std::ifstream ifsV(path + "/v.txt");
    std::ifstream ifsColor(path + "/color.txt");
    std::ifstream ifsNewFrame(path + "/NewFrame.txt");
    std::ifstream ifsResult(path + "/result.txt");

    //先读取结果
    std::string s;
    int lineCount = 0;

    while ( getline(ifsResult, s) )
    {
        std::vector<std::string> sVector;
        SplitString(s, sVector, " ");
        if (lineCount == 0)
        {
            stringstream ss;
            ss << sVector[0];
            ss >> pyrLevelsUsed;
        }
        else if (lineCount == 1)
        {
            init_T << stringTodouble(sVector[0]), stringTodouble(sVector[1]), stringTodouble(sVector[2]), stringTodouble(sVector[3]),
                   stringTodouble(sVector[4]), stringTodouble(sVector[5]), stringTodouble(sVector[6]), stringTodouble(sVector[7]),
                   stringTodouble(sVector[8]), stringTodouble(sVector[9]), stringTodouble(sVector[10]), stringTodouble(sVector[11]),
                   0, 0, 0, 1;
        }
        else if (lineCount == 2)
        {
            result_T << stringTodouble(sVector[0]), stringTodouble(sVector[1]), stringTodouble(sVector[2]), stringTodouble(sVector[3]),
                     stringTodouble(sVector[4]), stringTodouble(sVector[5]), stringTodouble(sVector[6]), stringTodouble(sVector[7]),
                     stringTodouble(sVector[8]), stringTodouble(sVector[9]), stringTodouble(sVector[10]), stringTodouble(sVector[11]),
                     0, 0, 0, 1;
        }
        else if (lineCount == 3)
        {
            ref_ab.a = stringTodouble(sVector[0]);
            ref_ab.b = stringTodouble(sVector[1]);
        }
        else if (lineCount == 4)
        {
            init_ab.a = stringTodouble(sVector[0]);
            init_ab.b = stringTodouble(sVector[1]);
        }
        else if (lineCount == 5)
        {
            result_ab.a = stringTodouble(sVector[0]);
            result_ab.b = stringTodouble(sVector[1]);
        }
        else if (lineCount == 6)
        {
            for (int i = 0; i < pyrLevelsUsed; i++)
                res[i] = stringTodouble(sVector[i]);
        }
        else if (lineCount == 7)
        {
            lastFlowIndicators[0] = stringTodouble(sVector[0]);
            lastFlowIndicators[1] = stringTodouble(sVector[1]);
            lastFlowIndicators[2] = stringTodouble(sVector[2]);
        }
        lineCount++;
    }

    lineCount=0;
    while ( getline(ifsK, s) )
    {
        std::vector<std::string> sVector;
        SplitString(s, sVector, " ");
        K[lineCount]<<stringTodouble(sVector[0]),0,stringTodouble(sVector[2]),
                0,stringTodouble(sVector[1]),stringTodouble(sVector[3]),
                0,0,1;

        lineCount++;
    }

    lineCount=0;
    while ( getline(ifsKi, s) )
    {
        std::vector<std::string> sVector;
        SplitString(s, sVector, " ");
        Ki[lineCount]<<stringTodouble(sVector[0]),0,stringTodouble(sVector[2]),
                0,stringTodouble(sVector[1]),stringTodouble(sVector[3]),
                0,0,1;

        lineCount++;
    }

    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
    {
        w[lvl] = ww >> lvl;
        h[lvl] = hh >> lvl;
        pc_u[lvl] = new double[w[lvl] * h[lvl]];
        pc_v[lvl] = new double[w[lvl] * h[lvl]];
        pc_idepth[lvl] = new double[w[lvl] * h[lvl]];
        pc_color[lvl] = new double[w[lvl] * h[lvl]];
        dINewl[lvl] = new Eigen::Vector3d[w[lvl] * h[lvl]];
    }

    //读取参考帧的点的深度
    int num = 0;
    int lvl = 0;
    while ( getline(ifsDepth, s) )
    {
        std::vector<std::string> sVector;
        SplitString(s, sVector, " ");
        num = sVector.size();
        pc_n[lvl] = sVector.size();
        for (int i = 0; i < num; i++)
        {
            pc_idepth[lvl][i] = stringTodouble(sVector[i]);
        }
        lvl++;
    }
    lvl = 0;
    while ( getline(ifsU, s) )
    {
        std::vector<std::string> sVector;
        SplitString(s, sVector, " ");
        num = sVector.size();
        for (int i = 0; i < num; i++)
        {
            pc_u[lvl][i] = stringTodouble(sVector[i]);
        }
        lvl++;
    }
    lvl = 0;
    while ( getline(ifsV, s) )
    {
        std::vector<std::string> sVector;
        SplitString(s, sVector, " ");
        num = sVector.size();
        for (int i = 0; i < num; i++)
        {
            pc_v[lvl][i] = stringTodouble(sVector[i]);
        }
        lvl++;
    }
    lvl = 0;
    while ( getline(ifsColor, s) )
    {
        std::vector<std::string> sVector;
        SplitString(s, sVector, " ");
        num = sVector.size();
        for (int i = 0; i < num; i++)
        {
            pc_color[lvl][i] = stringTodouble(sVector[i]);
        }
        lvl++;
    }

    lvl = 0;
    while ( getline(ifsNewFrame, s) )
    {
        std::vector<std::string> sVector;
        SplitString(s, sVector, " ");
        num = sVector.size();
        for (int i = 0; i < num; i += 3)
        {
            dINewl[lvl][i / 3][0] = stringTodouble(sVector[i]);
            dINewl[lvl][i / 3][1] = stringTodouble(sVector[i + 1]);
            dINewl[lvl][i / 3][2] = stringTodouble(sVector[i + 2]);
        }
        lvl++;
    }
}


int main(int argc, char** argv)
{
    //读取每一帧的数据
    std::string path = "/media/ren/99146341-07be-4601-9682-0539688db03f/fdso_tmp";

    int id = 8;

    //读取数据
    readData(path, id);

   //cout << init_T << endl;
//  cout << result_T << endl;
    //cout<<ref_ab.a<<ref_ab.b<<endl;
    //cout << init_ab.a <<" "<<init_ab.b<<endl;
//  cout << pyrLevelsUsed << endl;
//    cout<<pc_n[0]<<" "<<pc_n[1]<<" "<<pc_n[2]<<endl;
//    cout<<K[1]<<std::endl;
//     cout<<Ki[1]<<std::endl;

    //每一层最大迭代次数，越最上层，迭代次数越多
    int maxIterations[] = {10, 20, 50, 50, 50};

    Eigen::Matrix4d computeT=init_T;
    AffLight computeAB=init_ab;
    double para_Pose[6], para_AB[2];

    //从最上层开始
    for (int lvl = pyrLevelsUsed-1; lvl >= 0; lvl--)
    {
        bool firstSet = true;
        //numTermsInE=0;

        ceres::Problem problem;
        //构建loss_function,这里使用CauchyLoss
        ceres::LossFunction *  loss_function = new ceres::HuberLoss(setting_huberTH);

//        std::cout<<computeT<<endl;
//        std::cout<<computeAB.a<<" "<<computeAB.b<<endl;

        eigen2para(computeT, computeAB, para_Pose, para_AB);

//        ceres::LocalParameterization* local_parameterization = new LocalParameterizationSE3();
//        problem.AddParameterBlock(para_Pose, 6,local_parameterization);
//        problem.AddParameterBlock(para_AB, 2);

//        std::cout<<para_Pose[0]<<" "<<para_Pose[1]<<" "<<para_Pose[2]<<" "<<para_Pose[3]<<" "<<para_Pose[4]<<" "<<para_Pose[5]<<endl;
//        std::cout<<para_AB[0]<<" "<<para_AB[1]<<endl;
//        std::cout<<pc_n[lvl]<<endl;

        for (int i = 0; i < pc_n[lvl]; i++)
        {
            AlignmentFactor* f = new AlignmentFactor(dINewl[lvl], Eigen::Vector2d(pc_u[lvl][i], pc_v[lvl][i]),
                    pc_idepth[lvl][i], pc_color[lvl][i], ref_ab, K[lvl], Ki[lvl], w[lvl], h[lvl], 20.0);

            problem.AddResidualBlock(f, loss_function, para_Pose, para_AB);
        }

        //设为舒尔补
        // setting the solver
        ceres::Solver::Options options;
        //    options.max_solver_time_in_seconds = 5;
        //用于计算线性最小二乘问题的线性求解器的类型
        options.linear_solver_type = ceres::DENSE_SCHUR;
        //默认情况下，根据vlog级别，将最小化进程的过程记录到STDERR。
        //如果这个标志被设置为true，并且 Solver::Options::loggingtype不是 SILENT，日志输出将被发送到 STDOUT
        options.minimizer_progress_to_stdout = false;
        //用户指定的计算Jacobian 和残差的线程数量
        options.num_threads = 1;
        //在LINE_SEARCH和TRUST_REGION算法之间进行选择
        options.minimizer_type = ceres::TRUST_REGION;
        //Ceres使用的置信域计算方法，当前可选LEVENBERG_MARQUARDT和DOGLEG
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        //线性搜索终止条件
        //options.min_line_search_step_size=
        //options.dogleg_type = ceres::SUBSPACE_DOGLEG;
        //设置最大迭代次数
        options.max_num_iterations = maxIterations[lvl];
        options.num_linear_solver_threads = 1;
        //options.use_explicit_schur_complement = true;
        //options.use_nonmonotonic_steps = true;

        ceres::Solver::Summary summary;
        //开始优化
        ceres::Solve(options, &problem, &summary);

        //cout<<"numTermsInE: "<<numTermsInE<<endl;
        cout<<summary.BriefReport()<<endl;

        param2eigen(para_Pose,para_AB,computeT,computeAB);

//        std::cout<<computeT<<endl;
//        std::cout<<computeAB.a<<" "<<computeAB.b<<endl;
    }
}
