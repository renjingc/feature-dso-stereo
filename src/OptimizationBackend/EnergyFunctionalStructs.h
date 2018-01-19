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
#include "OptimizationBackend/RawResidualJacobian.h"

namespace fdso
{

class PointFrameResidual;
class CalibHessian;
class FrameHessian;
class PointHessian;

class EFResidual;
class EFPoint;
class EFFrame;
class EnergyFunctional;

/**
 * @brief      Class for ef residual.
 * 点与帧的残差
 */
class EFResidual
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	/**
	 * @brief      { function_description }
	 *
	 * @param      org      The organization
	 * @param      point_   The point
	 * @param      host_    The host
	 * @param      target_  The target
	 * 初始化残差
	 */
	inline EFResidual(PointFrameResidual* org, EFPoint* point_, EFFrame* host_, EFFrame* target_) :
		data(org), point(point_), host(host_), target(target_)
	{
		//初始为false
		isLinearized=false;
		isActiveAndIsGoodNEW=false;

		//新的残差雅克比
		J = new RawResidualJacobian();
		assert(((long)this)%16==0);
		assert(((long)J)%16==0);
	}
	inline ~EFResidual()
	{
		delete J;
	}

	void takeDataF();
	/**
	 * @brief      { function_description }
	 *
	 * @param      ef    { parameter_description }
	 * fix线性化当前ef
	 */
	void fixLinearizationF(EnergyFunctional* ef);

	// structural pointers
	//点与帧的残差
	PointFrameResidual* data;
	//主导帧的id,目标帧的id
	int hostIDX, targetIDX;

	//点
	EFPoint* point;
	//主导帧
	EFFrame* host;
	//目标帧
	EFFrame* target;
	int idxInAll;

	RawResidualJacobian* J;

	EIGEN_ALIGN16 VecNRf res_toZeroF;
	EIGEN_ALIGN16 Vec8f JpJdF;


	// status.
	//状态
	bool isLinearized;

	// if residual is not OOB & not OUTLIER & should be used during accumulations
	bool isActiveAndIsGoodNEW;
	inline const bool &isActive() const {return isActiveAndIsGoodNEW;}
};


enum EFPointStatus {PS_GOOD=0, PS_MARGINALIZE, PS_DROP};

/**
 * @brief      Class for ef point.
 * 点
 */
class EFPoint
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EFPoint(PointHessian* d, EFFrame* host_) : data(d),host(host_)
	{
		takeData();
		stateFlag=EFPointStatus::PS_GOOD;
	}
	void takeData();

	//点Hessian
	PointHessian* data;

	//先验priorF
	float priorF;
	//增量F
	float deltaF;

	// constant info (never changes in-between).
	//id
	int idxInPoints;
	//主导帧
	EFFrame* host;

	// contains all residuals.
	//该点的残差
	std::vector<EFResidual*> residualsAll;

	float bdSumF;
	float HdiF;
	float Hdd_accLF;
	VecCf Hcd_accLF;
	float bd_accLF;
	float Hdd_accAF;
	VecCf Hcd_accAF;
	float bd_accAF;

	//点的状态
	EFPointStatus stateFlag;
};

/**
 * @brief      Class for ef frame.
 * 帧
 */
class EFFrame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	EFFrame(std::shared_ptr<FrameHessian> d) : data(d)
	{
		takeData();
	}
	void takeData();

	//先验的Hessian矩阵
	Vec8 prior;				// prior hessian (diagonal)
	//与先验的偏差
	Vec8 delta_prior;		// = state-state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
	//与零状态的偏差
	Vec8 delta;				// state - state_zero.

	//包括的点
	std::vector<EFPoint*> points;

	//帧Hessian
	std::shared_ptr<FrameHessian> data;

	//窗口中帧idx
	int idx;	// idx in frames.

	//关键帧id
	int frameID;
};

}

