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

using namespace std;
using namespace cv;

/**
 * @brief The AlignmentFactor class
 *
 * 直接法误差因子，这里使用SizedCostFunstion
 * 残差为1维，优化变量为6,2维，共2个，i帧相对与参考帧位姿，a和b
 */
class AlignmentFactor : public ceres::SizedCostFunction<1,6,2>
{
public:
    AlignmentFactor() = delete ;
    AlignmentFactor(double fx, double fy, double cx, double cy, int w, int h)
            :fx(fx), fy(fy), cx(cx), cy(cy)
	{

	}
	bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
	{
        if ( jacobians )
        {
            if (jacobians[0])
            {

            }
            if (jacobians[1])
            {

            }
        }
    }
    void check(double **parameters)
    {

    }

    double fx, fy, cx, cy;
    int width;
    int height;
};

#define PYR_LEVELS 6

float* pc_u[PYR_LEVELS];
float* pc_v[PYR_LEVELS];
//参考帧的点逆深度，
float* pc_idepth[PYR_LEVELS];
//参考帧的点灰度值
float* pc_color[PYR_LEVELS];
//参考帧的点个数
int pc_n[PYR_LEVELS];

int pyrLevelsUsed = 0;

Eigen::Vector3f* dINewl[PYR_LEVELS];

Eigen::Matrix4f init_T, result_T;
float init_a, init_b, result_a, result_b;
Eigen::Matrix<double, 5, 1> res;
Eigen::Vector3d lastFlowIndicators;

int ww = 424;
int hh = 320;

int w[PYR_LEVELS],h[PYR_LEVELS];

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

float stringTofloat(string s)
{
	stringstream ss;
	float num;
	ss << s;
	ss >> num;
	return num;
}

void readData(string path, int id)
{
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
			init_T << stringTofloat(sVector[0]), stringTofloat(sVector[1]), stringTofloat(sVector[2]), stringTofloat(sVector[3]),
			       stringTofloat(sVector[4]), stringTofloat(sVector[5]), stringTofloat(sVector[6]), stringTofloat(sVector[7]),
			       stringTofloat(sVector[8]), stringTofloat(sVector[9]), stringTofloat(sVector[10]), stringTofloat(sVector[11]),
			       0, 0, 0, 1;
		}
		else if (lineCount == 2)
		{
			result_T << stringTofloat(sVector[0]), stringTofloat(sVector[1]), stringTofloat(sVector[2]), stringTofloat(sVector[3]),
			         stringTofloat(sVector[4]), stringTofloat(sVector[5]), stringTofloat(sVector[6]), stringTofloat(sVector[7]),
			         stringTofloat(sVector[8]), stringTofloat(sVector[9]), stringTofloat(sVector[10]), stringTofloat(sVector[11]),
			         0, 0, 0, 1;
		}
		else if (lineCount == 3)
		{
			init_a = stringTofloat(sVector[0]);
			init_b = stringTofloat(sVector[1]);
		}
		else if (lineCount == 4)
		{
			result_a = stringTofloat(sVector[0]);
			result_b = stringTofloat(sVector[1]);
		}
		else if (lineCount == 5)
		{
			for (int i = 0; i < pyrLevelsUsed; i++)
				res[i] = stringTofloat(sVector[i]);
		}
		else if (lineCount == 6)
		{
			lastFlowIndicators[0] = stringTofloat(sVector[0]);
			lastFlowIndicators[1] = stringTofloat(sVector[1]);
			lastFlowIndicators[2] = stringTofloat(sVector[2]);
		}
		lineCount++;
	}

	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
	{
        w[lvl] = ww >> lvl;
        h[lvl] = hh >> lvl;
        pc_u[lvl] = new float[w[lvl] * h[lvl]];
        pc_v[lvl] = new float[w[lvl] * h[lvl]];
        pc_idepth[lvl] = new float[w[lvl] * h[lvl]];
        pc_color[lvl] = new float[w[lvl] * h[lvl]];
        dINewl[lvl] = new Eigen::Vector3f[w[lvl] * h[lvl]];
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
			pc_idepth[lvl][i] = stringTofloat(sVector[i]);
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
			pc_u[lvl][i] = stringTofloat(sVector[i]);
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
			pc_v[lvl][i] = stringTofloat(sVector[i]);
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
			pc_color[lvl][i] = stringTofloat(sVector[i]);
		}
		lvl++;
	}

    lvl = 0;
    while ( getline(ifsNewFrame, s) )
    {
        std::vector<std::string> sVector;
        SplitString(s, sVector, " ");
        num = sVector.size();
        for (int i = 0; i < num; i+=3)
        {
            dINewl[lvl][i/3][0] = stringTofloat(sVector[i]);
            dINewl[lvl][i/3][1] = stringTofloat(sVector[i + 1]);
            dINewl[lvl][i/3][2] = stringTofloat(sVector[i + 2]);
        }
        lvl++;
    }
}


int main(int argc, char** argv)
{
	//读取每一帧的数据
	std::string path = "/media/ren/99146341-07be-4601-9682-0539688db03f/fdso_tmp";

	int id = 1;

	//读取数据
	readData(path, id);

//	cout << init_T << endl;
//	cout << result_T << endl;
//	cout << init_a << endl;
//	cout << init_b << endl;
//	cout << result_a << endl;
//	cout << result_b << endl;
//	cout << pyrLevelsUsed << endl;
//    cout<<pc_n[0]<<" "<<pc_n[1]<<" "<<pc_n[2]<<endl;

    //每一层最大迭代次数，越最上层，迭代次数越多
    int maxIterations[] = {10, 20, 50, 50, 50};
    //从最上层开始
    for (int lvl = pyrLevelsUsed; lvl >= 0; lvl--)
    {

    }
}
