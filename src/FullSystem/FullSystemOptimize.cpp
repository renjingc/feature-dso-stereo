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



#include "FullSystem/FullSystem.h"

#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/ResidualProjections.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include <cmath>

#include <algorithm>

namespace fdso
{

/**
 * @brief      { function_description }
 *
 * @param[in]  fixLinearization  The fix linearization
 * @param      toRemove          To remove
 * @param[in]  min               The minimum
 * @param[in]  max               The maximum
 * @param      stats             The statistics
 * @param[in]  tid               The tid
 * 
 * 全部线性化
 */
void FullSystem::linearizeAll_Reductor(bool fixLinearization, std::vector<PointFrameResidual*>* toRemove, int min, int max, Vec10* stats, int tid)
{
	//遍历每一个残差
	for (int k = min; k < max; k++)
	{
		//遍历每一个点与帧产生的残差，进行线性化，即计算点与帧之间的残差和各个雅克比
		PointFrameResidual* r = activeResiduals[k];
		(*stats)[0] += r->linearize(&Hcalib);

		if (fixLinearization)
		{
			//更新残差和状态state_energy = state_NewEnergy
			r->applyRes(true);

			if (r->efResidual->isActive())
			{
				if (r->isNew)
				{
					PointHessian* p = r->point;
					Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].PRE_KRKiTll * Vec3f(p->u, p->v, 1);	// projected point assuming infinite depth.
					Vec3f ptp = ptp_inf + r->host->targetPrecalc[r->target->idx].PRE_KtTll * p->idepth_scaled;	// projected point with real depth.
					float relBS = 0.01 * ((ptp_inf.head<2>() / ptp_inf[2]) - (ptp.head<2>() / ptp[2])).norm();	// 0.01 = one pixel.

					if (relBS > p->maxRelBaseline)
						p->maxRelBaseline = relBS;

					p->numGoodResiduals++;
				}
			}
			else
			{
				toRemove[tid].push_back(activeResiduals[k]);
			}
		}
	}
}

/**
 * @brief      { function_description }
 *
 * @param[in]  copyJacobians  The copy jacobians
 * @param[in]  min            The minimum
 * @param[in]  max            The maximum
 * @param      stats          The statistics
 * @param[in]  tid            The tid
 * 
 * 更新残差
 */
void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid)
{
	for (int k = min; k < max; k++)
		activeResiduals[k]->applyRes(true);
}

/**
 * @brief      Sets the new frame energy th.
 * 设置新的一帧的阈值frameEnergyTH
 */
void FullSystem::setNewFrameEnergyTH()
{

	// collect all residuals and make decision on TH.
	//残差数
	allResVec.clear();
	allResVec.reserve(activeResiduals.size() * 2);
	//最新的一帧
	FrameHessian* newFrame = frameHessians.back();

	//遍历全部的点和帧的残差
	for (PointFrameResidual* r : activeResiduals)
		if (r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame)
		{
			allResVec.push_back(r->state_NewEnergyWithOutlier);
		}

	if (allResVec.size() == 0)
	{
		newFrame->frameEnergyTH = 12 * 12 * patternNum;
		return;		// should never happen, but lets make sure.
	}


	int nthIdx = setting_frameEnergyTHN * allResVec.size();

	assert(nthIdx < (int)allResVec.size());
	assert(setting_frameEnergyTHN < 1);

	std::nth_element(allResVec.begin(), allResVec.begin() + nthIdx, allResVec.end());
	float nthElement = sqrtf(allResVec[nthIdx]);

	newFrame->frameEnergyTH = nthElement * setting_frameEnergyTHFacMedian;
	newFrame->frameEnergyTH = 26.0f * setting_frameEnergyTHConstWeight + newFrame->frameEnergyTH * (1 - setting_frameEnergyTHConstWeight);
	newFrame->frameEnergyTH = newFrame->frameEnergyTH * newFrame->frameEnergyTH;
	newFrame->frameEnergyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;

//
//	int good=0,bad=0;
//	for(float f : allResVec) if(f<newFrame->frameEnergyTH) good++; else bad++;
//	printf("EnergyTH: mean %f, median %f, result %f (in %d, out %d)! \n",
//			meanElement, nthElement, sqrtf(newFrame->frameEnergyTH),
//			good, bad);
}

/**
 * @brief      { function_description }
 *
 * @param[in]  fixLinearization  The fix linearization
 *
 * @return     { description_of_the_return_value }
 */
Vec3 FullSystem::linearizeAll(bool fixLinearization)
{
	double lastEnergyP = 0;
	//lastEnergyR没用
	double lastEnergyR = 0;
	double num = 0;

	//线程数
	std::vector<PointFrameResidual*> toRemove[NUM_THREADS];
	for (int i = 0; i < NUM_THREADS; i++) toRemove[i].clear();

	//多线程线性化，线性化全部的当前帧的点与帧之间的残差，即计算残差和雅克比
	if (multiThreading)
	{
		treadReduce.reduce(boost::bind(&FullSystem::linearizeAll_Reductor, this, fixLinearization, toRemove, _1, _2, _3, _4), 0, activeResiduals.size(), 0);
		lastEnergyP = treadReduce.stats[0];
	}
	else
	{
		Vec10 stats;
		linearizeAll_Reductor(fixLinearization, toRemove, 0, activeResiduals.size(), &stats, 0);
		lastEnergyP = stats[0];
	}

	//设置最新的关键帧的帧阈值
	setNewFrameEnergyTH();

	//是否fix
	if (fixLinearization)
	{
		//遍历每一个残差
		for (PointFrameResidual* r : activeResiduals)
		{
			PointHessian* ph = r->point;
			if (ph->lastResiduals[0].first == r)
				ph->lastResiduals[0].second = r->state_state;
			else if (ph->lastResiduals[1].first == r)
				ph->lastResiduals[1].second = r->state_state;
		}

		int nResRemoved = 0;
		for (int i = 0; i < NUM_THREADS; i++)
		{
			for (PointFrameResidual* r : toRemove[i])
			{
				PointHessian* ph = r->point;

				if (ph->lastResiduals[0].first == r)
					ph->lastResiduals[0].first = 0;
				else if (ph->lastResiduals[1].first == r)
					ph->lastResiduals[1].first = 0;

				for (unsigned int k = 0; k < ph->residuals.size(); k++)
					if (ph->residuals[k] == r)
					{
						ef->dropResidual(r->efResidual);
						deleteOut<PointFrameResidual>(ph->residuals, k);
						nResRemoved++;
						break;
					}
			}
		}
		//printf("FINAL LINEARIZATION: removed %d / %d residuals!\n", nResRemoved, (int)activeResiduals.size());
	}

	return Vec3(lastEnergyP, lastEnergyR, num);
}


// applies step to linearization point.
/**
 * @brief      { function_description }
 *
 * @param[in]  stepfacC  The stepfac c
 * @param[in]  stepfacT  The stepfac t
 * @param[in]  stepfacR  The stepfac r
 * @param[in]  stepfacA  The stepfac a
 * @param[in]  stepfacD  The stepfac d
 *
 * @return     { description_of_the_return_value }
 * 从备份的状态中进行下一步，判断是否可以跳出
 */
bool FullSystem::doStepFromBackup(float stepfacC, float stepfacT, float stepfacR, float stepfacA, float stepfacD)
{
//	float meanStepC=0,meanStepP=0,meanStepD=0;
//	meanStepC += Hcalib.step.norm();

	Vec10 pstepfac;
	pstepfac.segment<3>(0).setConstant(stepfacT);
	pstepfac.segment<3>(3).setConstant(stepfacR);
	pstepfac.segment<4>(6).setConstant(stepfacA);

	float sumA = 0, sumB = 0, sumT = 0, sumR = 0, sumID = 0, numID = 0;

	float sumNID = 0;

	if (setting_solverMode & SOLVER_MOMENTUM)
	{
		Hcalib.setValue(Hcalib.value_backup + Hcalib.step);
		for (FrameHessian* fh : frameHessians)
		{
			Vec10 step = fh->step;
			step.head<6>() += 0.5f * (fh->step_backup.head<6>());

			fh->setState(fh->state_backup + step);
			sumA += step[6] * step[6];
			sumB += step[7] * step[7];
			sumT += step.segment<3>(0).squaredNorm();
			sumR += step.segment<3>(3).squaredNorm();

			for (PointHessian* ph : fh->pointHessians)
			{
				float step = ph->step + 0.5f * (ph->step_backup);
				ph->setIdepth(ph->idepth_backup + step);
				sumID += step * step;
				sumNID += fabsf(ph->idepth_backup);
				numID++;

				ph->setIdepthZero(ph->idepth_backup + step);
			}
		}
	}
	else
	{
		Hcalib.setValue(Hcalib.value_backup + stepfacC * Hcalib.step);
		for (FrameHessian* fh : frameHessians)
		{
			fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
			sumA += fh->step[6] * fh->step[6];
			sumB += fh->step[7] * fh->step[7];
			sumT += fh->step.segment<3>(0).squaredNorm();
			sumR += fh->step.segment<3>(3).squaredNorm();

			for (PointHessian* ph : fh->pointHessians)
			{
				ph->setIdepth(ph->idepth_backup + stepfacD * ph->step);
				sumID += ph->step * ph->step;
				sumNID += fabsf(ph->idepth_backup);
				numID++;

				ph->setIdepthZero(ph->idepth_backup + stepfacD * ph->step);
			}
		}
	}

	sumA /= frameHessians.size();
	sumB /= frameHessians.size();
	sumR /= frameHessians.size();
	sumT /= frameHessians.size();
	sumID /= numID;
	sumNID /= numID;

	if (!setting_debugout_runquiet)
		printf("STEPS: A %.1f; B %.1f; R %.1f; T %.1f. \t",
		       sqrtf(sumA) / (0.0005 * setting_thOptIterations),
		       sqrtf(sumB) / (0.00005 * setting_thOptIterations),
		       sqrtf(sumR) / (0.00005 * setting_thOptIterations),
		       sqrtf(sumT)*sumNID / (0.00005 * setting_thOptIterations));


	EFDeltaValid = false;
	setPrecalcValues();

	return sqrtf(sumA) < 0.0005 * setting_thOptIterations &&
	       sqrtf(sumB) < 0.00005 * setting_thOptIterations &&
	       sqrtf(sumR) < 0.00005 * setting_thOptIterations &&
	       sqrtf(sumT) * sumNID < 0.00005 * setting_thOptIterations;
//
//	printf("mean steps: %f %f %f!\n",
//			meanStepC, meanStepP, meanStepD);
}

// sets linearization point.
/**
 * @brief      { function_description }
 *
 * @param[in]  backupLastStep  The backup last step
 × 备份最新的状态
 */
void FullSystem::backupState(bool backupLastStep)
{
	//备份矫正参数，备份帧状态，备份点的逆深度
	if (setting_solverMode & SOLVER_MOMENTUM)
	{
		if (backupLastStep)
		{
			Hcalib.step_backup = Hcalib.step;
			Hcalib.value_backup = Hcalib.value;
			for (FrameHessian* fh : frameHessians)
			{
				fh->step_backup = fh->step;
				fh->state_backup = fh->get_state();
				for (PointHessian* ph : fh->pointHessians)
				{
					ph->idepth_backup = ph->idepth;
					ph->step_backup = ph->step;
				}
			}
		}
		else
		{
			Hcalib.step_backup.setZero();
			Hcalib.value_backup = Hcalib.value;
			for (FrameHessian* fh : frameHessians)
			{
				fh->step_backup.setZero();
				fh->state_backup = fh->get_state();
				for (PointHessian* ph : fh->pointHessians)
				{
					ph->idepth_backup = ph->idepth;
					ph->step_backup = 0;
				}
			}
		}
	}
	else
	{
		Hcalib.value_backup = Hcalib.value;
		for (FrameHessian* fh : frameHessians)
		{
			fh->state_backup = fh->get_state();
			for (PointHessian* ph : fh->pointHessians)
				ph->idepth_backup = ph->idepth;
		}
	}
}

// sets linearization point.
/**
 * @brief      Loads a sate backup.
 * 加载之前的餐厨
 */
void FullSystem::loadSateBackup()
{
	//设置之前矫正参数
	Hcalib.setValue(Hcalib.value_backup);

	//遍历每一帧设置每一帧的状态
	for (FrameHessian* fh : frameHessians)
	{
		fh->setState(fh->state_backup);

		//遍历每一个点，设置每一个的逆深度和零状态逆深度
		for (PointHessian* ph : fh->pointHessians)
		{
			ph->setIdepth(ph->idepth_backup);
			ph->setIdepthZero(ph->idepth_backup);
		}

	}

	EFDeltaValid = false;

	//重新设置帧间关系
	setPrecalcValues();
}

/**
 * @brief      Calculates the m energy.
 *
 * @return     The m energy.
 */
double FullSystem::calcMEnergy()
{
	if (setting_forceAceptStep) return 0;
	// calculate (x-x0)^T * [2b + H * (x-x0)] for everything saved in L.
	//ef->makeIDX();
	//ef->setDeltaF(&Hcalib);
	return ef->calcMEnergyF();
}

/**
 * @brief      { function_description }
 *
 * @param[in]  res       The resource
 * @param[in]  resL      The resource l
 * @param[in]  resM      The resource m
 * @param[in]  resPrior  The resource prior
 * @param[in]  LExact    The l exact
 * @param[in]  a         { parameter_description }
 * @param[in]  b         { parameter_description }
 */
void FullSystem::printOptRes(Vec3 res, double resL, double resM, double resPrior, double LExact, float a, float b)
{
	printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n",
	       res[0],
	       sqrtf((float)(res[0] / (patternNum * ef->resInA))),
	       ef->resInA,
	       ef->resInM,
	       a,
	       b
	      );
}

/**
 * @brief
 *
 * @param[in]  mnumOptIts  The mnum option its  迭代次数
 *
 * @return     { description_of_the_return_value }
 */
float FullSystem::optimize(int mnumOptIts)
{

	if (frameHessians.size() < 2) return 0;
	if (frameHessians.size() < 3) mnumOptIts = 20;
	if (frameHessians.size() < 4) mnumOptIts = 15;

	// get statistics and active residuals.

	//全部的残差
	activeResiduals.clear();
	//点的个数
	int numPoints = 0;
	//有效的残差个数
	int numLRes = 0;

	//遍历，加入全部的残差
	for (FrameHessian* fh : frameHessians)
		for (PointHessian* ph : fh->pointHessians)
		{
			for (PointFrameResidual* r : ph->residuals)
			{
				if (!r->efResidual->isLinearized)
				{
					activeResiduals.push_back(r);
					r->resetOOB();
				}
				else
					numLRes++;
			}
			numPoints++;
		}

	if (!setting_debugout_runquiet)
		printf("OPTIMIZE %d pts, %d active res, %d lin res!\n", ef->nPoints, (int)activeResiduals.size(), numLRes);

	//线性化计算全部的点和帧的残差和雅克比
	Vec3 lastEnergy = linearizeAll(false);

	double lastEnergyL = calcLEnergy();
	double lastEnergyM = calcMEnergy();

	//多线程更新残差
	if (multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
	else
		applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);


	if (!setting_debugout_runquiet)
	{
		printf("Initial Error       \t");
		printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
	}

	//输出跟踪
	debugPlotTracking();

	//
	double lambda = 1e-1;
	float stepsize = 1;
	VecX previousX = VecX::Constant(CPARS + 8 * frameHessians.size(), NAN);

	//迭代
	for (int iteration = 0; iteration < mnumOptIts; iteration++)
	{
		// solve!
		//不是第一次迭代的时候，备份状态，即备份优化目标，设置带backup的优化目标
		backupState(iteration != 0);
		//solveSystemNew(0);

		//开始计算
		solveSystem(iteration, lambda);

		//改变的
		double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm());

		//之前的x
		previousX = ef->lastX;

		if (std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM))
		{
			float newStepsize = exp(incDirChange * 1.4);
			if (incDirChange < 0 && stepsize > 1) stepsize = 1;

			stepsize = sqrtf(sqrtf(newStepsize * stepsize * stepsize * stepsize));
			if (stepsize > 2) stepsize = 2;
			if (stepsize < 0.25) stepsize = 0.25;
		}

		//是否可以跳出
		bool canbreak = doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);

		// eval new energy!
		//新的残差和雅克比
		Vec3 newEnergy = linearizeAll(false);

		//计算新的误差
		double newEnergyL = calcLEnergy();
		double newEnergyM = calcMEnergy();

		if (!setting_debugout_runquiet)
		{
			printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",
			       (newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
			        lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) ? "ACCEPT" : "REJECT",
			       iteration,
			       log10(lambda),
			       incDirChange,
			       stepsize);
			printOptRes(newEnergy, newEnergyL, newEnergyM , 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
		}

		//更新残差
		if (setting_forceAceptStep || (newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
		                               lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM))
		{
			if (multiThreading)
				treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
			else
				applyRes_Reductor(true, 0, activeResiduals.size(), 0, 0);

			lastEnergy = newEnergy;
			lastEnergyL = newEnergyL;
			lastEnergyM = newEnergyM;

			lambda *= 0.25;
		}
		else
		{
			//加载上一次备份的状态
			loadSateBackup();
			lastEnergy = linearizeAll(false);
			lastEnergyL = calcLEnergy();
			lastEnergyM = calcMEnergy();
			lambda *= 1e2;
		}

		//超过迭代次数，跳出
		if (canbreak && iteration >= setting_minOptIterations) break;
	}

	//新的帧状态
	Vec10 newStateZero = Vec10::Zero();
	newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6);

	//设置帧新的状态，设置位姿和状态
	frameHessians.back()->setEvalPT(frameHessians.back()->PRE_worldToCam,
	                                newStateZero);
	EFDeltaValid = false;
	EFAdjointsValid = false;

	//
	ef->setAdjointsF(&Hcalib);

	//设置帧之间的关系
	setPrecalcValues();

	//最新的残差和雅克比
	lastEnergy = linearizeAll(true);

	if (!std::isfinite((double)lastEnergy[0]) || !std::isfinite((double)lastEnergy[1]) || !std::isfinite((double)lastEnergy[2]))
	{
		printf("KF Tracking failed: LOST!\n");
		isLost = true;
	}

	//最新的RMSE误差
	statistics_lastFineTrackRMSE = sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));

	if (calibLog != 0)
	{
		(*calibLog) << Hcalib.value_scaled.transpose() <<
		            " " << frameHessians.back()->get_state_scaled().transpose() <<
		            " " << sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA))) <<
		            " " << ef->resInM << "\n";
		calibLog->flush();
	}

	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		for (FrameHessian* fh : frameHessians)
		{
			fh->shell->camToWorld = fh->PRE_camToWorld;
			fh->shell->aff_g2l = fh->aff_g2l();
		}
	}

	debugPlotTracking();

	//emse误差
	return sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));
}

/**
 * @brief      { function_description }
 *
 * @param[in]  iteration  The iteration
 * @param[in]  lambda     The lambda
 */
void FullSystem::solveSystem(int iteration, double lambda)
{
	//获取零空间
	ef->lastNullspaces_forLogging = getNullspaces(
	                                    ef->lastNullspaces_pose,
	                                    ef->lastNullspaces_scale,
	                                    ef->lastNullspaces_affA,
	                                    ef->lastNullspaces_affB);

	//开始迭代计算
	ef->solveSystemF(iteration, lambda, &Hcalib);
}

/**
 * @brief      Calculates the l energy.
 *
 * @return     The l energy.
 */
double FullSystem::calcLEnergy()
{
	if (setting_forceAceptStep) return 0;

	double Ef = ef->calcLEnergyF_MT();
	return Ef;
}

/**
 * @brief      Removes outliers.
 * 根据点和帧是否有残差判断，加入pointHessiansOut，从pointHessians删除
 */
void FullSystem::removeOutliers()
{
	int numPointsDropped = 0;
	//遍历每一个关键帧
	for (FrameHessian* fh : frameHessians)
	{
		//遍历每一个点
		for (unsigned int i = 0; i < fh->pointHessians.size(); i++)
		{
			PointHessian* ph = fh->pointHessians[i];
			if (ph == 0) continue;

			//点的残差为0，则说明这个点与其它关键帧没有连接约束，说明该点可以踢了
			if (ph->residuals.size() == 0)
			{
				fh->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				fh->pointHessians[i] = fh->pointHessians.back();
				fh->pointHessians.pop_back();
				i--;
				numPointsDropped++;
			}
		}
	}
	//在ef误差函数中移除被边缘化的点，删除PS_DROP的点，即删除pointHessiansOut的点
	ef->dropPointsF();
}

/**
 * @brief      Gets the nullspaces.
 *
 * @param      nullspaces_pose   The nullspaces pose
 * @param      nullspaces_scale  The nullspaces scale
 * @param      nullspaces_affA   The nullspaces aff a
 * @param      nullspaces_affB   The nullspaces aff b
 *
 * @return     The nullspaces.
 * 获取零空间的位姿
 */
std::vector<VecX> FullSystem::getNullspaces(
    std::vector<VecX> &nullspaces_pose,
    std::vector<VecX> &nullspaces_scale,
    std::vector<VecX> &nullspaces_affA,
    std::vector<VecX> &nullspaces_affB)
{
	//先清空
	nullspaces_pose.clear();
	nullspaces_scale.clear();
	nullspaces_affA.clear();
	nullspaces_affB.clear();


	int n = CPARS + frameHessians.size() * 8;
	std::vector<VecX> nullspaces_x0_pre;
	for (int i = 0; i < 6; i++)
	{
		VecX nullspace_x0(n);
		nullspace_x0.setZero();
		for (FrameHessian* fh : frameHessians)
		{
			nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_pose.col(i);
			nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
			nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
		}
		nullspaces_x0_pre.push_back(nullspace_x0);
		nullspaces_pose.push_back(nullspace_x0);
	}
	for (int i = 0; i < 2; i++)
	{
		VecX nullspace_x0(n);
		nullspace_x0.setZero();
		for (FrameHessian* fh : frameHessians)
		{
			nullspace_x0.segment<2>(CPARS + fh->idx * 8 + 6) = fh->nullspaces_affine.col(i).head<2>();
			nullspace_x0[CPARS + fh->idx * 8 + 6] *= SCALE_A_INVERSE;
			nullspace_x0[CPARS + fh->idx * 8 + 7] *= SCALE_B_INVERSE;
		}
		nullspaces_x0_pre.push_back(nullspace_x0);
		if (i == 0) nullspaces_affA.push_back(nullspace_x0);
		if (i == 1) nullspaces_affB.push_back(nullspace_x0);
	}

	VecX nullspace_x0(n);
	nullspace_x0.setZero();
	for (FrameHessian* fh : frameHessians)
	{
		nullspace_x0.segment<6>(CPARS + fh->idx * 8) = fh->nullspaces_scale;
		nullspace_x0.segment<3>(CPARS + fh->idx * 8) *= SCALE_XI_TRANS_INVERSE;
		nullspace_x0.segment<3>(CPARS + fh->idx * 8 + 3) *= SCALE_XI_ROT_INVERSE;
	}
	nullspaces_x0_pre.push_back(nullspace_x0);
	nullspaces_scale.push_back(nullspace_x0);

	return nullspaces_x0_pre;
}

}
