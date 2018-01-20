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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/OpenCV/ImageDisplay.h"
#include "util/globalCalib.h"

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/ImmaturePoint.h"
#include "math.h"

namespace fdso
{

/**
 * @brief      { function_description }
 *
 * @param      point      The point
 * @param[in]  minObs     The minimum obs
 * @param      residuals  The residuals
 *
 * @return     { description_of_the_return_value }
 * 
 * 优化每一个点，将这个点与窗口中的每一个非主导帧的关键帧进行误差迭代，获取该点最新的逆深度
 * 从ImmaturePoint生成PointHessian
 */
std::shared_ptr<PointHessian> FullSystem::optimizeImmaturePoint(
		std::shared_ptr<ImmaturePoint> point, int minObs,
		ImmaturePointTemporaryResidual* residuals)
{
	int nres = 0;

	//遍历窗口中的每一帧
	for(std::shared_ptr<FrameHessian> fh : frameHessians)
	{
		//该点的主导帧不是该帧
		if(fh != point->host)
		{
			//设置残差和状态和目标帧
			residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
			residuals[nres].state_NewState = ResState::OUTLIER;
			residuals[nres].state_state = ResState::IN;
			residuals[nres].target = fh;
			nres++;
		}
	}
	assert(nres == ((int)frameHessians.size())-1);

	bool print = false;//rand()%50==0;

	//最新的残差和跟新Ｈ和ｂ
	float lastEnergy = 0;
	float lastHdd=0;
	float lastbd=0;

	//当前点逆深度＝最大逆深度和最小逆深度的１/2
	float currentIdepth=(point->idepth_max+point->idepth_min)*0.5f;

	//遍历每一种目标帧
	for(int i=0;i<nres;i++)
	{
		//该点线性化残差
		lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals+i,lastHdd, lastbd, currentIdepth);

		//更新残差和状态
		residuals[i].state_state = residuals[i].state_NewState;
		residuals[i].state_energy = residuals[i].state_NewEnergy;
	}

	//若全部的残差值和lastHdd小于阈值100,则说明该点没有很好的约束连接
	if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
	{
		if(print)
			printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
				nres, lastHdd, lastEnergy);
		return 0;
	}

	//激活的点，即有约束的点
	if(print) printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n" ,
			nres, lastHdd,lastEnergy,currentIdepth);

	float lambda = 0.1;
	//迭代３次
	for(int iteration=0;iteration<setting_GNItsOnPointActivation;iteration++)
	{
		//初始Ｈ
		float H = lastHdd;
		H *= 1+lambda;
		float step = (1.0/H) * lastbd;

		//新的逆深度
		float newIdepth = currentIdepth - step;

		//
		float newHdd=0; float newbd=0; float newEnergy=0;

		//将每一次误差相加
		for(int i=0;i<nres;i++)
			newEnergy += point->linearizeResidual(&Hcalib, 1, residuals+i,newHdd, newbd, newIdepth);

		if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)
		{
			if(print) printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
					nres,
					newHdd,
					lastEnergy);
			return 0;
		}

		if(print) printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
				(true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
				iteration,
				log10(lambda),
				"",
				lastEnergy, newEnergy, newIdepth);

		//新的误差小于上一时刻的误差
		if(newEnergy < lastEnergy)
		{
			//更新最新的逆深度
			currentIdepth = newIdepth;
			//更新最新的Ｈ,b和残差
			lastHdd = newHdd;
			lastbd = newbd;
			lastEnergy = newEnergy;

			//更新与每一帧的残差和状态
			for(int i=0;i<nres;i++)
			{
				residuals[i].state_state = residuals[i].state_NewState;
				residuals[i].state_energy = residuals[i].state_NewEnergy;
			}

			//步进缩小
			lambda *= 0.5;
		}
		else
		{
			lambda *= 5;
		}

		//当步进很小了，则跳出
		if(fabsf(step) < 0.0001*currentIdepth)
			break;
	}

	//逆深度错误
	if(!std::isfinite(currentIdepth))
	{
		printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
		return nullptr;//(std::shared_ptr<PointHessian>)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}

	//当前点与这一帧有好的残差
	int numGoodRes=0;
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN) numGoodRes++;

	//好的约束小于阈值，则该点也该out
	if(numGoodRes < minObs)
	{
		if(print) printf("OptPoint: OUTLIER!\n");
		return nullptr;//(std::shared_ptr<PointHessian>)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
	}

	//新建该点Hessian
	std::shared_ptr<PointHessian> p(new PointHessian(point, &Hcalib));
	if(!std::isfinite(p->energyTH)) 
	{
		//delete p;
		return nullptr;
	}//(std::shared_ptr<PointHessian>)((long)(-1));}

	//设置该点的逆深度和状态
	p->lastResiduals[0].first = 0;
	p->lastResiduals[0].second = ResState::OOB;
	p->lastResiduals[1].first = 0;
	p->lastResiduals[1].second = ResState::OOB;
	p->setIdepthZero(currentIdepth);
	p->setIdepth(currentIdepth);
	p->setPointStatus(PointHessian::ACTIVE);

	//遍历与之相关每一帧
	for(int i=0;i<nres;i++)
		if(residuals[i].state_state == ResState::IN)
		{
			//创建该点与每一个帧的残差
			PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target);
			r->state_NewEnergy = r->state_energy = 0;
			r->state_NewState = ResState::OUTLIER;
			r->setState(ResState::IN);

			//当前点插入这个残差
			p->residuals.push_back(r);

			//若当前的目标帧为最新的一帧
			if(r->target == frameHessians.back())
			{
				p->lastResiduals[0].first = r;
				p->lastResiduals[0].second = ResState::IN;
			}
			//若当前窗口中小于２帧，该帧的目标帧＝＝０
			//若当前窗口大于２帧，该帧的目标的帧为前面的面一帧
			else if(r->target == (frameHessians.size()<2 ? 0 : frameHessians[frameHessians.size()-2]))
			{
				p->lastResiduals[1].first = r;
				p->lastResiduals[1].second = ResState::IN;
			}
		}

	if(print) printf("point activated!\n");

	//激活的点个数++
	statistics_numActivatedPoints++;
	return p;
}

}
