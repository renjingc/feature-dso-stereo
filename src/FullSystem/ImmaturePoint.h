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

#include "FullSystem/FrameHessian.h"
#include "FullSystem/PointHessian.h"
#include "FullSystem/CalibHessian.h"
#include "FeatureDetector.h"

namespace fdso
{

/**
 * 点残差状态
 */
struct ImmaturePointTemporaryResidual
{
public:
	//残差状态
	ResState state_state;
	double state_energy;
	//新的残差状态
	ResState state_NewState;
	double state_NewEnergy;

	//主导帧
	std::shared_ptr<FrameHessian> target;
};

/**
 * 未成熟点的
 * IPS_OOB
 * IPS_OUTLIER都是不用到了的，IPS_OUTLIER更严重，像素坐标已经不对了
 *
 * IPS_SKIPPED和IPS_BADCONDITION都是迭代范围过小了的，像素坐标还是可以用的
 *
 * IPS_UNINITIALIZED初始状态
 */
enum ImmaturePointStatus {
	IPS_GOOD = 0,					// traced well and good     跟踪好的点
	IPS_OOB,					// OOB: end tracking & marginalize!	停止跟踪需要边缘化的点,超出图像的点
	IPS_OUTLIER,				// energy too high: if happens again: outlier!		计算的误差过大，需要剔除的点，逆深度值不好的点
	IPS_SKIPPED,				// traced well and good (but not actually traced).	跟踪好的，但最大和最小逆深度算出来的像素坐标差距过小，不进行迭代
	IPS_BADCONDITION,			// not traced because of bad condition.		最大逆深度值不好了，且像素误差大于距离了
	IPS_UNINITIALIZED 			// not even traced once.					一次都没跟踪的点，初始的状态
};


class ImmaturePoint
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// static values
	// 颜色，每个点都有8个
	float color[MAX_RES_PER_POINT];
	//每个点每个模式的权重
	float weights[MAX_RES_PER_POINT];

	//该点的xy方向的梯度
	Mat22f gradH;
	//???没用到
	Vec2f gradH_ev;
	//???没用到
	Mat22f gradH_eig;

	//误差能量阈值
	float energyTH;
	//像素坐标，在主导帧上
	float u, v;
	float u_stereo, v_stereo;  // u, v used to do static stereo matching
	//主导帧的Hessian矩阵
	std::shared_ptr<FrameHessian> host;
	//在未成熟的点中的id
	int idxInImmaturePoints;

	//质量
	float quality;
	//类型
	float my_type;

	//最小的逆深度
	float idepth_min;
	//最大的逆深度
	float idepth_max;
	float idepth_min_stereo;  // idepth_min used to do static matching
	float idepth_max_stereo;  // idepth_max used to do static matching
	//双目得到的逆深度
	float idepth_stereo;

	Feature* mF=nullptr;
	int feaMode=0;

	//初始化
	ImmaturePoint(int u_, int v_, std::shared_ptr<FrameHessian> host_, float type, CalibHessian* HCalib);
	ImmaturePoint(float u_, float v_, std::shared_ptr<FrameHessian> host_, CalibHessian* HCalib);
	~ImmaturePoint();

	//双目静态匹配
	ImmaturePointStatus traceStereo(std::shared_ptr<FrameHessian> frame, Mat33f K, bool mode_right);
	//深度跟踪
	ImmaturePointStatus traceOn(std::shared_ptr<FrameHessian> frame, Mat33f hostToFrame_KRKi, Vec3f hostToFrame_Kt, Vec2f hostToFrame_affine, CalibHessian* HCalib, bool debugPrint = false);

	//最新的跟踪状态
	ImmaturePointStatus lastTraceStatus;
	//最新帧的像素坐标
	Vec2f lastTraceUV;
	//最新帧的像素间隔
	float lastTracePixelInterval;

	//点的残差线性化
	double linearizeResidual(
	  CalibHessian *  HCalib, const float outlierTHSlack,
	  ImmaturePointTemporaryResidual* tmpRes,
	  float &Hdd, float &bd,
	  float idepth);

	//获取点的梯度
	float getdPixdd(
	  CalibHessian *  HCalib,
	  ImmaturePointTemporaryResidual* tmpRes,
	  float idepth);

	//计算残差
	float calcResidual(
	  CalibHessian *  HCalib, const float outlierTHSlack,
	  ImmaturePointTemporaryResidual* tmpRes,
	  float idepth);

private:
};

}

