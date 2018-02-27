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
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/OutputWrapper/Output3DWrapper.h"

#include "FullSystem/CoarseTracker.h"

namespace fdso
{

/**
 * @brief      { function_description }
 *
 * @param      newFH  The new fh
 * 判断窗口中的哪些帧该被边缘化
 * １. 前帧的点个数过小，则该帧被边缘化或者该帧与最新的帧的光度变化较大，且剩下的帧数大于最小帧数
 *  2. 帧数大于最大帧数，则移除与其它帧距离和最大的一帧
 */
void FullSystem::flagFramesForMarginalization(FrameHessian* newFH)
{
	if (setting_minFrameAge > setting_maxFrames)
	{
		for (int i = setting_maxFrames; i < (int)frameHessians.size(); i++)
		{
			FrameHessian* fh = frameHessians[i - setting_maxFrames];
			LOG(INFO) << "frame " << fh->frameID << " is set as marged" << endl;
			fh->flaggedForMarginalization = true;

			Frame* f1;
			Frame* tempF(new Frame());
			tempF->id = fh->shell->id;
			//获取所有关键帧
			auto allKFs = globalMap->getAllKFs();

			std::set<Frame*, CmpFrameID>::iterator iter;
			iter = allKFs.find(tempF);
			if (iter != allKFs.end())
			{
				f1 = (*iter);
				//遍历所有关键帧
				f1->update = false;
			}
			delete tempF;
		}
		return;
	}

	//边缘化个数
	int flagged = 0;
	// marginalize all frames that have not enough points.
	//遍历每一帧，当前帧的点个数过小，则该帧被边缘化
	for (int i = 0; i < (int)frameHessians.size(); i++)
	{
		FrameHessian* fh = frameHessians[i];
		//内点个数
		int in = fh->pointHessians.size() + fh->immaturePoints.size();
		//out点个数
		int out = fh->pointHessiansMarginalized.size() + fh->pointHessiansOut.size();

		//与最新的帧的光度变化
		Vec2 refToFh = AffLight::fromToVecExposure(frameHessians.back()->ab_exposure, fh->ab_exposure,
		               frameHessians.back()->aff_g2l(), fh->aff_g2l());

		//内点个数比例小于0.05或者a变化过大，logs(a)>0.7
		//剩下的帧数大于最小帧数，５
		if ( (in < setting_minPointsRemaining * (in + out) || fabs(logf((float)refToFh[0])) > setting_maxLogAffFacInWindow)
		     && ((int)frameHessians.size()) - flagged > setting_minFrames)
		{
//			printf("MARGINALIZE frame %d, as only %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
//					fh->frameID, in, in+out,
//					(int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
//					(int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
//					visInLast, outInLast,
//					fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
			//这一帧设为被边缘化
			LOG(INFO) << "frame " << fh->frameID << " is set as marged" << endl;
			fh->flaggedForMarginalization = true;
			flagged++;

			Frame* f1;
			Frame* tempF(new Frame());
			tempF->id = fh->shell->id;
			//获取所有关键帧
			auto allKFs = globalMap->getAllKFs();

			std::set<Frame*, CmpFrameID>::iterator iter;
			iter = allKFs.find(tempF);
			if (iter != allKFs.end())
			{
				f1 = (*iter);
				//遍历所有关键帧
				f1->update = false;
			}
			delete tempF;
		}
		else
		{
//			printf("May Keep frame %d, as %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
//					fh->frameID, in, in+out,
//					(int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
//					(int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
//					visInLast, outInLast,
//					fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
		}
	}

	// marginalize one.
	//剩下的帧大于最大帧数
	if ((int)frameHessians.size() - flagged >= setting_maxFrames)
	{
		double smallestScore = 1;
		FrameHessian* toMarginalize = 0;
		//最新的帧
		FrameHessian* latest = frameHessians.back();

		//遍历每一帧
		for (FrameHessian* fh : frameHessians)
		{
			//最新帧跳过
			if (fh->frameID > latest->frameID - setting_minFrameAge || fh->frameID == 0) continue;
			//if(fh==frameHessians.front() == 0) continue;

			double distScore = 0;
			//当前帧与目标帧的
			for (FrameFramePrecalc &ffh : fh->targetPrecalc)
			{
				if (ffh.target->frameID > latest->frameID - setting_minFrameAge + 1 || ffh.target == ffh.host) continue;
				//距离和
				distScore += 1 / (1e-5 + ffh.distanceLL);
			}
			//
			distScore *= -sqrtf(fh->targetPrecalc.back().distanceLL);

			//距离值最小的，即该帧与其他帧的距离最大的，该帧设为移除
			if (distScore < smallestScore)
			{
				smallestScore = distScore;
				toMarginalize = fh;
			}
		}

//		printf("MARGINALIZE frame %d, as it is the closest (score %.2f)!\n",
//				toMarginalize->frameID, smallestScore);
		//
		LOG(INFO) << "frame " << toMarginalize->frameID << " is set as marged" << endl;
		toMarginalize->flaggedForMarginalization = true;
		flagged++;

		Frame* f1;
		Frame* tempF(new Frame());
		tempF->id = toMarginalize->shell->id;
		//获取所有关键帧
		auto allKFs = globalMap->getAllKFs();

		std::set<Frame*, CmpFrameID>::iterator iter;
		iter = allKFs.find(tempF);
		if (iter != allKFs.end())
		{
			f1 = (*iter);
			//遍历所有关键帧
			f1->update = false;
		}
		delete tempF;
	}

//	printf("FRAMES LEFT: ");
//	for(FrameHessian* fh : frameHessians)
//		printf("%d ", fh->frameID);
//	printf("\n");
}

/**
 * @brief      { function_description }
 *
 * @param      frame  The frame
 * 进行边缘化
 */
void FullSystem::marginalizeFrame(FrameHessian* frame)
{
	// marginalize or remove all this frames points.

	assert((int)frame->pointHessians.size() == 0);

	//在ef误差函数中边缘化该帧
	ef->marginalizeFrame(frame->efFrame);

	// drop all observations of existing points in that frame.

	//遍历每一帧
	for (FrameHessian* fh : frameHessians)
	{
		if (fh == frame) continue;

		//遍历每一个点
		for (PointHessian* ph : fh->pointHessians)
		{
			//遍历每个残差
			for (unsigned int i = 0; i < ph->residuals.size(); i++)
			{
				PointFrameResidual* r = ph->residuals[i];
				//移除目标帧为该点的残差
				if (r->target == frame)
				{
					if (ph->lastResiduals[0].first == r)
						ph->lastResiduals[0].first = 0;
					else if (ph->lastResiduals[1].first == r)
						ph->lastResiduals[1].first = 0;

					//debug
					if (r->host->frameID < r->target->frameID)
						statistics_numForceDroppedResFwd++;
					else
						statistics_numForceDroppedResBwd++;

					//在ef误差函数中移除被边缘化的点
					ef->dropResidual(r->efResidual);
					deleteOut<PointFrameResidual>(ph->residuals, i);
					break;
				}
			}
		}
	}


	{
		std::vector<FrameHessian*> v;
		v.push_back(frame);
		for (IOWrap::Output3DWrapper* ow : outputWrapper)
			ow->publishKeyframes(v, true, &Hcalib);
	}

	//该帧在哪一时刻被边缘化
	frame->shell->marginalizedAt = frameHessians.back()->shell->id;
	frame->shell->movedByOpt = frame->w2c_leftEps().norm();

	LOG(INFO) << "frame " << frame->frameID << " frame marged" << endl;
	//从队列中删除该帧

	// Frame* tempF(new Frame());
	// tempF->id = frame->shell->id;

	// //获取所有关键帧
	// auto allKFs = globalMap->getAllKFs();

	// std::set<Frame*, CmpFrameID>::iterator iter;
	// iter = allKFs.find(tempF);
	// if (iter != allKFs.end())
	// {
	// 	//找到要删除的这一帧的特征的指针,构造新的特征
	// 	std::vector<Feature*> temPFeaLst;
	// 	for(auto &fea : (*iter)->_features)
	// 	{
	// 		if(fea->_status==Feature::ACTIVE_IDEPTH)
	// 			fea->idepth=fea->mPH->idepth;

	// 		Feature* tmpFea(new Feature(*fea));
	// 		temPFeaLst.push_back(tmpFea);
	// 	}
	// 	//清空特征,放入新的特征
	// 	(*iter)->_features.clear();
	// 	for(auto & fea: temPFeaLst)
	// 	{
	// 		(*iter)->_features.push_back(fea);
	// 	}
	// }
	// else
	// {
	// 	cout << "Cannot find the Frame!" << endl;
	// }

	// delete tempF;

	for (unsigned int i = 0; i < frame-> _features.size(); i++)
	{
		frame->_features[i]->mImP = nullptr;
		frame->_features[i]->mPH = nullptr;
	}

	for (unsigned int i = 0; i < frame->rightFrame->_features.size(); i++)
	{
		frame->rightFrame->_features[i]->mImP = nullptr;
		frame->rightFrame->_features[i]->mPH = nullptr;
	}

	deleteOutOrder<FrameHessian>(frameHessians, frame);
	deleteOutOrder<FrameHessian>(frameHessiansRight, frame->rightFrame);
	// deleteOutOrder<FrameHessian>(frameHessiansRight, frame->rightFrame);

	//重置每一帧的idx
	for (unsigned int i = 0; i < frameHessians.size(); i++)
		frameHessians[i]->idx = i;

	for (unsigned int i = 0; i < frameHessiansRight.size(); i++)
		frameHessiansRight[i]->idx = i;

	//重新设置每一帧之间的关系
	setPrecalcValues();

	//调整F
	ef->setAdjointsF(&Hcalib);
}

}
