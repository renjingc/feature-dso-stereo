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
#include "vector"
#include <math.h>
#include "util/settings.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"

#include <string>
#include <sstream>
#include <fstream>
#include <sys/stat.h> 

namespace fdso
{
struct CalibHessian;
struct FrameHessian;
struct PointFrameResidual;

/**
 * 
 */
class CoarseTracker {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CoarseTracker(int w, int h);
	~CoarseTracker();

	void saveResult(
    		SE3 lastToNew_In, AffLight aff_g2l_In,
    		SE3 lastToNew_out, AffLight aff_g2l_out,
    		int coarsestLvl,
    		Vec5 minResForAbort);

    void saveK();

	//跟踪新一帧
	bool trackNewestCoarse(
			FrameHessian* newFrameHessian,
			SE3 &lastToNew_out, AffLight &aff_g2l_out,
			int coarsestLvl, Vec5 minResForAbort,
			IOWrap::Output3DWrapper* wrap=0);

	//设置第一帧的参考帧
	void setCTRefForFirstFrame(
			std::vector<FrameHessian*> frameHessians);

	//设置跟踪的参考帧
	void setCoarseTrackingRef(
			std::vector<FrameHessian*> frameHessians, FrameHessian* fh_right, CalibHessian Hcalib);

	//得到第一帧的深度图
	void makeCoarseDepthForFirstFrame(FrameHessian* fh);

	//设置内参
	void makeK(
			CalibHessian* HCalib);

	bool debugPrint, debugPlot;

	//内参
	Mat33f K[PYR_LEVELS];
	Mat33f Ki[PYR_LEVELS];
	float fx[PYR_LEVELS];
	float fy[PYR_LEVELS];
	float fxi[PYR_LEVELS];
	float fyi[PYR_LEVELS];
	float cx[PYR_LEVELS];
	float cy[PYR_LEVELS];
	float cxi[PYR_LEVELS];
	float cyi[PYR_LEVELS];
	int w[PYR_LEVELS];
	int h[PYR_LEVELS];

   	void debugPlotIDepthMap(float* minID, float* maxID, std::vector<IOWrap::Output3DWrapper*> &wraps);
    	void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps);

    	//最新跟踪的参考帧的Hessian矩阵
	FrameHessian* lastRef;
	//当前帧与参考帧的光度线性变换
	AffLight lastRef_aff_g2l;
	//新的一帧
	FrameHessian* newFrame;
	//参考帧的ID
	int refFrameID;

	// act as pure ouptut
	//最新的残差
	Vec5 lastResiduals;
	//三位，resOld从第3位开始后的三位
	Vec3 lastFlowIndicators;
	//第一次输出rmse
	double firstCoarseRMSE;
private:

	void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians, FrameHessian* fh_right, CalibHessian Hcalib);

	//按图的坐标来
	//逆深度图,
	float* idepth[PYR_LEVELS];
	//权重图
	float* weightSums[PYR_LEVELS];
	//上一时刻权重图
	float* weightSums_bak[PYR_LEVELS];

	//计算残差，更新H和b矩阵，这个函数没实现？
	Vec6 calcResAndGS(int lvl, Mat88 &H_out, Vec8 &b_out, SE3 refToNew, AffLight aff_g2l, float cutoffTH);
	//计算残差
	Vec6 calcRes(int lvl, SE3 refToNew, AffLight aff_g2l, float cutoffTH);
	//更新H和b矩阵
	void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, SE3 refToNew, AffLight aff_g2l);
	//更新H和b矩阵，这个函数没实现？
	void calcGS(int lvl, Mat88 &H_out, Vec8 &b_out, SE3 refToNew, AffLight aff_g2l);

	// pc buffers
	// 按有效的一个个顺序递增
	// 参考帧的点像素坐标
	float* pc_u[PYR_LEVELS];
	float* pc_v[PYR_LEVELS];
	//参考帧的点逆深度，
	float* pc_idepth[PYR_LEVELS];
	//参考帧的点灰度值
	float* pc_color[PYR_LEVELS];
	//参考帧的点个数
	int pc_n[PYR_LEVELS];

	// warped buffers
	// 变换后的逆深度
	float* buf_warped_idepth;
	//变换后的图像坐标
	float* buf_warped_u;
	float* buf_warped_v;
	//变换后的梯度
	float* buf_warped_dx;
	float* buf_warped_dy;
	//变换后的残差
	float* buf_warped_residual;
	//变换后的权重
	float* buf_warped_weight;
	//变换后的参考
	float* buf_warped_refColor;
	//变换后的
	int buf_warped_n;

	//Hessian矩阵
	Accumulator9 acc;
};

/**
 * 距离图
 */
class CoarseDistanceMap {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	CoarseDistanceMap(int w, int h);
	~CoarseDistanceMap();

	//创建距离图，一堆关键帧，和当前关键帧
	void makeDistanceMap(
			std::vector<FrameHessian*> frameHessians,
			FrameHessian* frame);

	//内点投票
	void makeInlierVotes(
			std::vector<FrameHessian*> frameHessians);

	void makeK( CalibHessian* HCalib);

	//最后距离值
	float* fwdWarpedIDDistFinal;

	//内参
	Mat33f K[PYR_LEVELS];
	Mat33f Ki[PYR_LEVELS];
	float fx[PYR_LEVELS];
	float fy[PYR_LEVELS];
	float fxi[PYR_LEVELS];
	float fyi[PYR_LEVELS];
	float cx[PYR_LEVELS];
	float cy[PYR_LEVELS];
	float cxi[PYR_LEVELS];
	float cyi[PYR_LEVELS];
	int w[PYR_LEVELS];
	int h[PYR_LEVELS];

	void addIntoDistFinal(int u, int v);
private:
	//残差
	PointFrameResidual** coarseProjectionGrid;
	int* coarseProjectionGridNum;
	Eigen::Vector2i* bfsList1;
	Eigen::Vector2i* bfsList2;

	void growDistBFS(int bfsNum);
};

}

