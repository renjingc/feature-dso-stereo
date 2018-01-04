#include "OptimizerPnP.h"

#include <g2o/g2o/core/block_solver.h>
#include <g2o/g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/g2o/core/robust_kernel_impl.h>
#include <g2o/g2o/solvers/linear_solver_eigen.h>
#include <g2o/g2o/solvers/linear_solver_dense.h>
#include <g2o/g2o/types/types_six_dof_expmap.h>
#include <g2o/g2o/types/types_seven_dof_expmap.h>

#include <Eigen/StdVector>

#include "Converter.h"

#include <mutex>

int PoseOptimization(const cv::Mat& mTcw, const std::vector<cv::KeyPoint>& mvKeys,
                                           const std::vector<cv::KeyPoint>& mvKeysRight,const std::vector<cv::KeyPoint>& mvKeysUn,
                                           const std::vector<cv::Point3f>& mvpMapPoints,
                                           const std::vector<float>& mvInvLevelSigma2,
                                           const float fx,const float fy,const float cx,const float cy,
                                           std::vector<bool>& mvbOutlier,
                                           cv::Mat& pose)
{
    // 该优化函数主要用于Tracking线程中：运动跟踪、参考帧跟踪、地图跟踪、重定位

    // 步骤1：构造g2o优化器
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType * linearSolver;

    linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();

    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    int nInitialCorrespondences=0;

    // Set Frame vertex
    // 步骤2：添加顶点：待优化当前帧的Tcw
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(Converter::toSE3Quat(mTcw));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    const int N = pFrame->N;

    // for Monocular
    vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    // for Stereo
    vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    // 步骤3：添加一元边：相机投影模型
    {
    // unique_lock<mutex> lock(MapPoint::mGlobalMutex);

    for(int i=0; i<N; i++)
    {
            cv::Point3f pMP = mvpMapPoints[i];
        // if(pMP)
        // {
            // Monocular observation
            // 单目情况, 也有可能在双目下, 当前帧的左兴趣点找不到匹配的右兴趣点
            if(mvuRight.size()<=0)
            {
                nInitialCorrespondences++;
                mvbOutlier[i] = false;

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = mvKeysUn[i];
                obs << kpUn.pt.x, kpUn.pt.y;

                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);

                const float invSigma2 = mvInvLevelSigma2[kpUn.octave];
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = fx;
                e->fy = fy;
                e->cx = cx;
                e->cy = cy;

                e->Xw[0] = pMP.x;
                e->Xw[1] = pMP.y;
                e->Xw[2] = pMP.z;

                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else  // Stereo observation 双目
            {
                nInitialCorrespondences++;
                mvbOutlier[i] = false;

                //SET EDGE
                Eigen::Matrix<double,3,1> obs;// 这里和单目不同
                const cv::KeyPoint &kpUn = mvKeysUn[i];
                const float &kp_ur = mvuRight[i];
                obs << kpUn.pt.x, kpUn.pt.y, kp_ur;// 这里和单目不同

                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();// 这里和单目不同

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(obs);

                const float invSigma2 = mvInvLevelSigma2[kpUn.octave];
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = fx;
                e->fy = fy;
                e->cx = cx;
                e->cy = cy;
                e->bf = mbf;

                e->Xw[0] = pMP.x;
                e->Xw[1] = pMP.y;
                e->Xw[2] = pMP.z;

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        // }
    }
    }


    if(nInitialCorrespondences<3)
        return 0;

    // We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    // At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    // 步骤4：开始优化，总共优化四次，每次优化后，将观测分为outlier和inlier，outlier不参与下次优化
    // 由于每次优化后是对所有的观测进行outlier和inlier判别，因此之前被判别为outlier有可能变成inlier，反之亦然
    // 基于卡方检验计算出的阈值（假设测量有一个像素的偏差）
    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};// 四次迭代，每次迭代的次数

    int nBad=0;
    for(size_t it=0; it<4; it++)
    {
        vSE3->setEstimate(Converter::toSE3Quat(mTcw));
        optimizer.initializeOptimization(0);// 对level为0的边进行优化
        optimizer.optimize(its[it]);

        nBad=0;
        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];

            if(mvbOutlier[idx])
            {
                e->computeError(); // NOTE g2o只会计算active edge的误差
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {
                mvbOutlier[idx]=true;
                e->setLevel(1);                 // 设置为outlier
                nBad++;
            }
            else
            {
                mvbOutlier[idx]=false;
                e->setLevel(0);                 // 设置为inlier
            }

            if(it==2)
                e->setRobustKernel(0); // 除了前两次优化需要RobustKernel以外, 其余的优化都不需要
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];

            if(mvbOutlier[idx])
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                mvbOutlier[idx]=true;
                e->setLevel(1);
                nBad++;
            }
            else
            {
                e->setLevel(0);
                mvbOutlier[idx]=false;
            }

            if(it==2)
                e->setRobustKernel(0);
        }

        if(optimizer.edges().size()<10)
            break;
    }
    // Recover optimized pose and return number of inliers
    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    pose = Converter::toCvMat(SE3quat_recov);
    // pFrame->SetPose(pose);

    return nInitialCorrespondences-nBad;
}
