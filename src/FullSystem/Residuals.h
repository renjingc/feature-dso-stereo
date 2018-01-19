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


#include "util/globalCalib.h"
#include "vector"

#include "util/NumType.h"
#include <iostream>
#include <fstream>
#include "util/globalFuncs.h"
#include "OptimizationBackend/RawResidualJacobian.h"

namespace fdso
{
class PointHessian;
class FrameHessian;
class CalibHessian;

class EFResidual;

/**
 *  当前帧残差处于模式，激活，线性化，边缘化
 */
enum ResLocation {ACTIVE = 0, LINEARIZED, MARGINALIZED, NONE};
//残差状态
enum ResState {IN = 0, OOB, OUTLIER};

/**
 * @brief      { struct_description }
 */
struct FullJacRowT
{
	Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
};

/**
 * @brief      Class for point frame residual.
 */
class PointFrameResidual
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	//残差
	EFResidual* efResidual;

	//计数
	static int instanceCounter;

	//残差状态
	ResState state_state;
	double state_energy;
	ResState state_NewState;
	double state_NewEnergy;
	double state_NewEnergyWithOutlier;

	//设置残差状态
	void setState(ResState s) {state_state = s;}

	//点的Hessian矩阵
	PointHessian* point;
	//主导帧
	std::shared_ptr<FrameHessian> host;
	//参考帧
	std::shared_ptr<FrameHessian> target;

	//原始的残差雅克比
	RawResidualJacobian* J;

	//是否是新的
	bool isNew;
	//重投影
	Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];

	//投影到目标帧的像素坐标和逆深度
	Vec3f centerProjectedTo;

	~PointFrameResidual();
	PointFrameResidual();
	PointFrameResidual(PointHessian* point_, std::shared_ptr<FrameHessian> host_, std::shared_ptr<FrameHessian> target_);
	double linearize(CalibHessian* HCalib);

	//重置
	void resetOOB()
	{
		//能量值＝０
		state_NewEnergy = state_energy = 0;
		//最新的状态为outlier
		state_NewState = ResState::OUTLIER;

		//设置初始状态位ＩＮ
		setState(ResState::IN);
	};
	void applyRes( bool copyJacobians);

	void debugPlot();

	void printRows(std::vector<VecX> &v, VecX &r, int nFrames, int nPoints, int M, int res);
};
}

