/**
* This file is part of DSO.
*
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

#include "util/NumType.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/OutputWrapper/Output3DWrapper.h"
#include "util/settings.h"
#include "vector"
#include <math.h>




namespace fdso
{
class CalibHessian;
class FrameHessian;


class Pnt
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// index in jacobian. never changes (actually, there is no reason why).
	//雅克比矩阵中点的坐标
	float u, v;

	// idepth / isgood / energy during optimization.
	//点的深度
	float idepth;
	//点是否好
	bool isGood;
	//当前点的误差能量值，第一项位光度误差的能量，第二项为正则项的能量
	Vec2f energy;		// (UenergyPhotometric, energyRegularizer)
	//新的是否是好
	bool isGood_new;
	//新的深度值
	float idepth_new;
	//当前点新的误差能量值
	Vec2f energy_new;

	float iR;
	float iRSumNum;

	//最新的Hessian
	float lastHessian;
	float lastHessian_new;

	// max stepsize for idepth (corresponding to max. movement in pixel-space).
	//用于计算深度的最大优化步骤
	float maxstep;

	// idx (x+y*w) of closest point one pyramid level above.
	//上一层最近的点
	int parent;
	//与上一层最近点的距离
	float parentDist;

	// idx (x+y*w) of up to 10 nearest points in pixel space.
	//点附近是个点
	int neighbours[10];
	//点附近10个点的距离
	float neighboursDist[10];

	float my_type;
	//离群值距离
	float outlierTH;
};

//用于初始化的类
class CoarseInitializer {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	CoarseInitializer(int w, int h);
	~CoarseInitializer();

	//设置第一帧的矫正Hessian,新一帧的Hessian
	void setFirst(CalibHessian* HCalib, FrameHessian* newFrameHessian);
	//跟踪帧
	void setFirstStereo(CalibHessian* HCalib, FrameHessian* newFrameHessian, FrameHessian* newFrameHessian_Right);
	bool trackFrame(FrameHessian* newFrameHessian, FrameHessian* newFrameHessian_Right, std::vector<IOWrap::Output3DWrapper*> &wraps);
	//计算梯度
	void calcTGrads(FrameHessian* newFrameHessian);

	//关键帧的ID号
	int frameID;
	//是否使用光学线性逆变换
	bool fixAffine;
	//是否输出调试信息
	bool printDebug;

	//每层的点的信息
	Pnt* points[PYR_LEVELS];
	int numPoints[PYR_LEVELS];
	//光学线性逆变换
	AffLight thisToNext_aff;
	//到下一时刻的变换矩阵
	SE3 thisToNext;
	SE3 T_WC_ini; // the pose of first cam0 frame.

	//第一帧的Hessian
	FrameHessian* firstFrame;
	//新一帧的Hessian
	FrameHessian* newFrame;

	FrameHessian* firstRightFrame;

private:
	Mat33 K[PYR_LEVELS];
	Mat33 Ki[PYR_LEVELS];
	double fx[PYR_LEVELS];
	double fy[PYR_LEVELS];
	double fxi[PYR_LEVELS];
	double fyi[PYR_LEVELS];
	double cx[PYR_LEVELS];
	double cy[PYR_LEVELS];
	double cxi[PYR_LEVELS];
	double cyi[PYR_LEVELS];
	int w[PYR_LEVELS];
	int h[PYR_LEVELS];
	void makeK(CalibHessian* HCalib);
	float* idepth[PYR_LEVELS];

	bool snapped;
	int snappedAt;

	// pyramid images & levels on all levels
	Eigen::Vector3f* dINew[PYR_LEVELS];
	Eigen::Vector3f* dIFist[PYR_LEVELS];

	Eigen::DiagonalMatrix<float, 8> wM;	 //权重对角线矩阵

	// temporary buffers for H and b.
	// 0-7存储雅克比和，(dd*dp),   第八项为sum(res*dd)，第9项为inverse hessian entry，权重
	Vec10f* JbBuffer;			// 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
	Vec10f* JbBuffer_new;

	Accumulator9 acc9;		 //前面8*8为Hessian矩阵，最后一列为b
	Accumulator9 acc9SC;	//前面8*8为Hessian矩阵，最后一列为b

	Vec3f dGrads[PYR_LEVELS];

	//权重参数
	float alphaK;
	float alphaW;
	//当前点的深度为当前这个点的深度与中值点深度互补,时中值点深度权重
	float regWeight;
	float couplingWeight;

	//计算残差
	Vec3f calcResAndGS(
	  int lvl,
	  Mat88f &H_out, Vec8f &b_out,
	  Mat88f &H_out_sc, Vec8f &b_out_sc,
	  SE3 refToNew, AffLight refToNew_aff,
	  bool plot);
      	//计算能量函数
	Vec3f calcEC(int lvl); // returns OLD NERGY, NEW ENERGY, NUM TERMS.
	void optReg(int lvl);

	void propagateUp(int srcLvl);
	void propagateDown(int srcLvl);
	float rescale();

	void resetPoints(int lvl);
	void doStep(int lvl, float lambda, Vec8f inc);
	void applyStep(int lvl);

	void makeGradients(Eigen::Vector3f** data);

	void debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps);
	void makeNN();
};

/**
 * 点云的FLANN
 * 最近邻搜索
 */
class FLANNPointcloud
{
public:
	inline FLANNPointcloud() {num = 0; points = 0;}
	inline FLANNPointcloud(int n, Pnt* p) :  num(n), points(p) {}
       	//点云个数
	int num;
       	//点
	Pnt* points;
	//获取kdtree树中点的个数
	inline size_t kdtree_get_point_count() const { return num; }
	//kdtree距离
	inline float kdtree_distance(const float *p1, const size_t idx_p2, size_t /*size*/) const
	{
		const float d0 = p1[0] - points[idx_p2].u;
		const float d1 = p1[1] - points[idx_p2].v;
		return d0 * d0 + d1 * d1;
	}

	inline float kdtree_get_pt(const size_t idx, int dim) const
	{
		if (dim == 0) return points[idx].u;
		else return points[idx].v;
	}
	template <class BBOX>
	bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

}


