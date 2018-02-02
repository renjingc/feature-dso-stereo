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
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/OutputWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>
#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>


namespace fdso
{
int FrameHessian::instanceCounter = 0;
int PointHessian::instanceCounter = 0;
int CalibHessian::instanceCounter = 0;

/**
 *
 */
FullSystem::FullSystem(std::shared_ptr<ORBVocabulary> voc):
	_vocab(voc)
{
	LOG(INFO) << "FullSystem Init" << std::endl;
	int retstat = 0;
	if (setting_logStuff)
	{
		retstat += system("rm -rf logs");
		retstat += system("mkdir logs");

		retstat += system("rm -rf mats");
		retstat += system("mkdir mats");

		calibLog = new std::ofstream();
		calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
		calibLog->precision(12);

		numsLog = new std::ofstream();
		numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
		numsLog->precision(10);

		coarseTrackingLog = new std::ofstream();
		coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
		coarseTrackingLog->precision(10);

		eigenAllLog = new std::ofstream();
		eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
		eigenAllLog->precision(10);

		eigenPLog = new std::ofstream();
		eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
		eigenPLog->precision(10);

		eigenALog = new std::ofstream();
		eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
		eigenALog->precision(10);

		DiagonalLog = new std::ofstream();
		DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
		DiagonalLog->precision(10);

		variancesLog = new std::ofstream();
		variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
		variancesLog->precision(10);


		nullspacesLog = new std::ofstream();
		nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
		nullspacesLog->precision(10);
	}
	else
	{
		nullspacesLog = 0;
		variancesLog = 0;
		DiagonalLog = 0;
		eigenALog = 0;
		eigenPLog = 0;
		eigenAllLog = 0;
		numsLog = 0;
		calibLog = 0;
	}

	assert(retstat != 293847);

	selectionMap = new float[wG[0]*hG[0]];

	coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
	coarseTracker = new CoarseTracker(wG[0], hG[0]);
	coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
	coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	pixelSelector = new PixelSelector(wG[0], hG[0]);

	statistics_lastNumOptIts = 0;
	statistics_numDroppedPoints = 0;
	statistics_numActivatedPoints = 0;
	statistics_numCreatedPoints = 0;
	statistics_numForceDroppedResBwd = 0;
	statistics_numForceDroppedResFwd = 0;
	statistics_numMargResFwd = 0;
	statistics_numMargResBwd = 0;

	lastCoarseRMSE.setConstant(100);

	currentMinActDist = 2;
	initialized = false;

	ef = new EnergyFunctional();
	ef->red = &this->treadReduce;

	isLost = false;
	initFailed = false;

	needNewKFAfter = -1;

	linearizeOperation = true;
	runMapping = true;
	mappingThread = boost::thread(&FullSystem::mappingLoop, this);
	lastRefStopID = 0;

	minIdJetVisDebug = -1;
	maxIdJetVisDebug = -1;
	minIdJetVisTracker = -1;
	maxIdJetVisTracker = -1;

	detectorLeft = new FeatureDetector(wG[0], hG[0], 10, 15.0);
	detectorRight = new FeatureDetector(wG[0], hG[0], 10, 15.0);

	matcher = new FeatureMatcher(65, 100, 30, 100, 0.7);
	globalMap = new Map();
	loopClosing = new LoopClosing(this);
}

/**
 *
 */
FullSystem::~FullSystem()
{
	blockUntilMappingIsFinished();

	if (setting_logStuff)
	{
		calibLog->close(); delete calibLog;
		numsLog->close(); delete numsLog;
		coarseTrackingLog->close(); delete coarseTrackingLog;
		//errorsLog->close(); delete errorsLog;
		eigenAllLog->close(); delete eigenAllLog;
		eigenPLog->close(); delete eigenPLog;
		eigenALog->close(); delete eigenALog;
		DiagonalLog->close(); delete DiagonalLog;
		variancesLog->close(); delete variancesLog;
		nullspacesLog->close(); delete nullspacesLog;
	}

	delete[] selectionMap;

	for (FrameShell* s : allFrameHistory)
		delete s;

	for (FrameHessian* s : frameHessians)
		delete s;

	for (FrameShell* s : allKeyFramesHistory)
		delete s;

	for (FrameHessian* fh : unmappedTrackedFrames)
		delete fh;

	for (FrameHessian* fh : unmappedTrackedFrames_right)
		delete fh;

	delete coarseDistanceMap;
	delete coarseTracker;
	delete coarseTracker_forNewKF;
	delete coarseInitializer;
	delete pixelSelector;
	delete ef;
	delete globalMap;
}

/**
 * [FullSystem::setOriginalCalib description]
 * @param originalCalib [description]
 * @param originalW     [description]
 * @param originalH     [description]
 */
void FullSystem::setOriginalCalib(VecXf originalCalib, int originalW, int originalH)
{
}

/**
 * [FullSystem::setGammaFunction description]
 * @param BInv [description]
 */
void FullSystem::setGammaFunction(float* BInv)
{
	if (BInv == 0) return;

	// copy BInv.
	memcpy(Hcalib.Binv, BInv, sizeof(float) * 256);

	// invert.
	for (int i = 1; i < 255; i++)
	{
		// find val, such that Binv[val] = i.
		// I dont care about speed for this, so do it the stupid way.

		for (int s = 1; s < 255; s++)
		{
			if (BInv[s] <= i && BInv[s + 1] >= i)
			{
				Hcalib.B[i] = s + (i - BInv[s]) / (BInv[s + 1] - BInv[s]);
				break;
			}
		}
	}
	Hcalib.B[0] = 0;
	Hcalib.B[255] = 255;
}

/**
 * [FullSystem::printResult description]
 * @param file [description]
 */
void FullSystem::printResult(std::string file)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	std::ofstream myfile;
	myfile.open (file.c_str());
	myfile << std::setprecision(15);
	int i = 0;

	Eigen::Matrix<double, 3, 3> last_R = (*(allFrameHistory.begin()))->camToWorldOpti.so3().matrix();
	Eigen::Matrix<double, 3, 1> last_T = (*(allFrameHistory.begin()))->camToWorldOpti.translation().transpose();

	for (FrameShell* s : allFrameHistory)
	{
		if (!s->poseValid)
		{
			myfile << last_R(0, 0) << " " << last_R(0, 1) << " " << last_R(0, 2) << " " << last_T(0, 0) << " " <<
			       last_R(1, 0) << " " << last_R(1, 1) << " " << last_R(1, 2) << " " << last_T(1, 0) << " " <<
			       last_R(2, 0) << " " << last_R(2, 1) << " " << last_R(2, 2) << " " << last_T(2, 0) << "\n";
			continue;
		}

		if (setting_onlyLogKFPoses && s->marginalizedAt == s->id)
		{
			myfile << last_R(0, 0) << " " << last_R(0, 1) << " " << last_R(0, 2) << " " << last_T(0, 0) << " " <<
			       last_R(1, 0) << " " << last_R(1, 1) << " " << last_R(1, 2) << " " << last_T(1, 0) << " " <<
			       last_R(2, 0) << " " << last_R(2, 1) << " " << last_R(2, 2) << " " << last_T(2, 0) << "\n";
			continue;
		}

		const Eigen::Matrix<double, 3, 3> R = s->camToWorld.so3().matrix();
		const Eigen::Matrix<double, 3, 1> T = s->camToWorld.translation().transpose();

		last_R = R;
		last_T = T;

		myfile << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << " " << T(0, 0) << " " <<
		       R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << " " << T(1, 0) << " " <<
		       R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << " " << T(2, 0) << "\n";

//		myfile << s->timestamp <<
//			" " << s->camToWorld.translation().transpose()<<
//			" " << s->camToWorld.so3().unit_quaternion().x()<<
//			" " << s->camToWorld.so3().unit_quaternion().y()<<
//			" " << s->camToWorld.so3().unit_quaternion().z()<<
//			" " << s->camToWorld.so3().unit_quaternion().w() << "\n";
		i++;
	}
	myfile.close();
}

/**
 * [FullSystem::trackNewCoarse description]
 * @param  fh       [description]
 * @param  fhRight [description]
 * @return          [description]
 */
Vec4 FullSystem::trackNewCoarse(FrameHessian* fh, FrameHessian* fhRight, SE3 initT, bool usePnP)
{

	assert(allFrameHistory.size() > 0);
	// set pose initialization.

	// show original images
	//发布双目原始图
	for (IOWrap::Output3DWrapper* ow : outputWrapper)
	{
		ow->pushStereoLiveFrame(fh, fhRight);
	}

	// Sophus::SO3 init_R(R);
	// Eigen::Vector3d t(0, 0, 0); //
	// Sophus::SE3 init_Rt(R, t); //

	//参考帧
	FrameHessian* lastF = coarseTracker->lastRef;

	//a和b
	AffLight aff_last_2_l = AffLight(0, 0);

	//
	std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;

	// for first two frames process differently
	if (allFrameHistory.size() == 2)
	{
		LOG(INFO) << "init track" << std::endl;
		initializeFromInitializerStereo(fh);

//		lastF_2_fh_tries.push_back(init_Rt);
		lastF_2_fh_tries.push_back(SE3(Eigen::Matrix<double, 3, 3>::Identity(), Eigen::Matrix<double, 3, 1>::Zero() ));

		for (float rotDelta = 0.02; rotDelta < 0.05; rotDelta = rotDelta + 0.02)
		{
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0, 0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0, 0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0, 0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0, 0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0, 0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0, 0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
		}

		//设置内参
		coarseTracker->makeK(&Hcalib);
		//设置第一帧的参考帧,设置参考帧的逆深度图和每个点的权重pc_u，pc_v，pc_idepth，pc_color
		//权重图weightSums
		coarseTracker->setCTRefForFirstFrame(frameHessians);

		//最新的参考帧
		lastF = coarseTracker->lastRef;
	}
	else
	{
		LOG(INFO) << "tracking" << std::endl;
		//上一帧
		FrameShell* slast = allFrameHistory[allFrameHistory.size() - 2];
		//上上一帧
		FrameShell* sprelast = allFrameHistory[allFrameHistory.size() - 3];
		SE3 slast_2_sprelast;
		SE3 lastF_2_slast;
		{	// lock on global pose consistency!
			boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
			//上上一帧相对与上一帧的相对位姿
			slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
			//上一帧相对与参考帧的位姿的变换
			lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
			//上一帧的a和b变换
			aff_last_2_l = slast->aff_g2l;
		}

		//上上一帧
		SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.


		// get last delta-movement.
		//lastF_2_fh_tries.push_back(init_Rt.inverse()*lastF_2_slast);
		//匀速模型，上一次的位移*上一帧相对参考帧的位姿
		if (usePnP)
		{
			SE3 T_tmp = lastF->shell->camToWorld * initT.inverse();
			//平移
			Eigen::Matrix<double, 3, 1> last_T = T_tmp.translation().transpose();
			// std::cout << "pnp pose  x:" << last_T(0, 0) << "y:" << last_T(1, 0) << "z:" << last_T(2, 0) << std::endl;
			lastF_2_fh_tries.push_back(initT);
		}

		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	// assume constant motion.
		//两次运动
		lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
		//一半的运动
		lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse() * lastF_2_slast); // assume half motion.
		//无运动
		lastF_2_fh_tries.push_back(lastF_2_slast); // assume zero motion.
		//与关键帧无运动
		lastF_2_fh_tries.push_back(SE3()); // assume zero motion FROM KF.

		/*        lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);

		        	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);
		        	lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() *  SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);

		        	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);
		        	lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() *  SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);

		        	lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);
		        	lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() *  SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);*/

		// just try a TON of different initializations (all rotations). In the end,
		// if they don't work they will only be tried on the coarsest level, which is super fast anyway.
		// also, if tracking rails here we loose, so we really, really want to avoid that.
		// 匀速模型*角度的噪声
		for (float rotDelta = 0.02; rotDelta < 0.02; rotDelta++)
		{
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, 0), Vec3(0, 0, 0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, 0), Vec3(0, 0, 0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, rotDelta), Vec3(0, 0, 0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, 0), Vec3(0, 0, 0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, 0), Vec3(0, 0, 0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, 0, -rotDelta), Vec3(0, 0, 0)));			// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, 0), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, 0), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, 0), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, 0, -rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, 0, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, -rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, -rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, -rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
			lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1, rotDelta, rotDelta, rotDelta), Vec3(0, 0, 0)));	// assume constant motion.
		}

		if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid)
		{
			lastF_2_fh_tries.clear();
			lastF_2_fh_tries.push_back(SE3());
		}
	}

	Vec3 flowVecs = Vec3(100, 100, 100);
	SE3 lastF_2_fh = SE3();
	AffLight aff_g2l = AffLight(0, 0);

	// as long as maxResForImmediateAccept is not reached, I'll continue through the options.
	// I'll keep track of the so-far best achieved residual for each level in achievedRes.
	// If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.

	//用于更新最好的残差
	Vec5 achievedRes = Vec5::Constant(NAN);

	//是否一次好的寻找
	bool haveOneGood = false;

	//寻找次数
	int tryIterations = 0;

	//一般都是第一个位姿就迭代成功了
	for (unsigned int i = 0; i < lastF_2_fh_tries.size(); i++)
	{
		AffLight aff_g2l_this = aff_last_2_l;
		SE3 lastF_2_fh_this = lastF_2_fh_tries[i];

		//是否跟踪成功
		bool trackingIsGood;
		if (usePnP && i == 0)
			trackingIsGood = coarseTracker->trackNewestCoarse(
			                   fh, lastF_2_fh_this, aff_g2l_this,
			                   pyrLevelsUsed - 2,
			                   achievedRes);	// in each level has to be at least as good as the last try.
		else
			trackingIsGood = coarseTracker->trackNewestCoarse(
			                   fh, lastF_2_fh_this, aff_g2l_this,
			                   pyrLevelsUsed - 1,
			                   achievedRes);	// in each level has to be at least as good as the last try.
		//尝试次数++
		tryIterations++;

		if (i != 0)
		{
			LOG(INFO) << "RE-TRACK ATTEMPT " << i << " with initOption " << i << " and  start-lvl " << pyrLevelsUsed - 1
			          << "ab: " << aff_g2l_this.a << " " << aff_g2l_this.b << " " <<
			          achievedRes[0] << " " <<
			          achievedRes[1] << " " <<
			          achievedRes[2] << " " <<
			          achievedRes[3] << " " <<
			          achievedRes[4] << " -> " <<
			          coarseTracker->lastResiduals[0] << " " <<
			          coarseTracker->lastResiduals[1] << " " <<
			          coarseTracker->lastResiduals[2] << " " <<
			          coarseTracker->lastResiduals[3] << " " <<
			          coarseTracker->lastResiduals[4] << std::endl;
		}

		// do we have a new winner?
		// 跟踪成功,残差是否有值,lastResiduals每一层的残差，lastResiduals[0]就是第一层的残差
		if (trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) && !(coarseTracker->lastResiduals[0] >=  achievedRes[0]))
		{
			//printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
			flowVecs = coarseTracker->lastFlowIndicators;
			aff_g2l = aff_g2l_this;
			lastF_2_fh = lastF_2_fh_this;
			//成功
			haveOneGood = true;
		}

		// take over achieved res (always).
		// 更新achievedRes
		if (haveOneGood)
		{
			for (int i = 0; i < 5; i++)
			{
				//残差减小了，更新每一层的残差
				if (!std::isfinite((float)achievedRes[i]) || achievedRes[i] > coarseTracker->lastResiduals[i])	// take over if achievedRes is either bigger or NAN.
					achievedRes[i] = coarseTracker->lastResiduals[i];
			}
		}

		//成功了，且残差小于一定阈值,setting_reTrackThreshold=1.5
		if (haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
			break;
	}

	//不成功，则跟踪失败
	if (!haveOneGood)
	{
		LOG(INFO) << "BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover" << std::endl;;
		flowVecs = Vec3(0, 0, 0);
		aff_g2l = aff_last_2_l;
		lastF_2_fh = lastF_2_fh_tries[0];
	}

	//每一层的残差，即记录上一次的残差
	lastCoarseRMSE = achievedRes;

	// no lock required, as fh is not used anywhere yet.
	// 更新位姿
	fh->shell->camToTrackingRef = lastF_2_fh.inverse();
	fh->shell->trackingRef = lastF->shell;
	fh->shell->aff_g2l = aff_g2l;
	fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

	//平移
	Eigen::Matrix<double, 3, 1> last_T = fh->shell->camToWorld.translation().transpose();
	// std::cout << "dso pose  x:" << last_T(0, 0) << "y:" << last_T(1, 0) << "z:" << last_T(2, 0) << std::endl;
	LOG(INFO) << "pose  x:" << last_T(0, 0) << "y:" << last_T(1, 0) << "z:" << last_T(2, 0) << std::endl;

	if (coarseTracker->firstCoarseRMSE < 0)
		coarseTracker->firstCoarseRMSE = achievedRes[0];

	LOG(INFO) << "Coarse Tracker tracked ab = " << aff_g2l.a << " " << aff_g2l.b << " " << fh->ab_exposure << " " << achievedRes[0] << std::endl;

	if (setting_logStuff)
	{
		(*coarseTrackingLog) << std::setprecision(16)
		                     << fh->shell->id << " "
		                     << fh->shell->timestamp << " "
		                     << fh->ab_exposure << " "
		                     << fh->shell->camToWorld.log().transpose() << " "
		                     << aff_g2l.a << " "
		                     << aff_g2l.b << " "
		                     << achievedRes[0] << " "
		                     << tryIterations << "\n";
	}

	//返回残差第一个值，从第三位后的三位
	//1：平移后的像素重投影误差误差/个数/2
	//2： 0
	//3：平移旋转后像素重投影误差误差/个数/2
	//std::cout<<achievedRes[0]<<" "<<flowVecs[0]<<" "<<flowVecs[1]<<" "<<flowVecs[2]<<std::endl;
	return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

void FullSystem::stereoMatch(FrameHessian* fh, FrameHessian* fhRight)
{

	Mat33f K = Mat33f::Identity();
	K(0, 0) = Hcalib.fxl();
	K(1, 1) = Hcalib.fyl();
	K(0, 2) = Hcalib.cxl();
	K(1, 2) = Hcalib.cyl();

//    std::vector<cv::KeyPoint> keypoints_left;
//    std::vector<cv::KeyPoint> keypoints_right;
//    std::vector<cv::DMatch> matches;
//    float debugKeepPercentage = 0.2;

	for (ImmaturePoint* ip : fh->immaturePoints)
	{
		ip->u_stereo = ip->u;
		ip->v_stereo = ip->v;
		ip->idepth_min_stereo = ip->idepth_min = 0;
		ip->idepth_max_stereo = ip->idepth_max = NAN;


//      std::cout << "idx: " << ip->idxInImmaturePoints << "\t Right." << std::endl;
		ImmaturePointStatus phTraceRightStatus = ip->traceStereo(fhRight, K, 1);

		if (phTraceRightStatus == ImmaturePointStatus::IPS_GOOD)
		{
			ImmaturePoint* ipRight(new ImmaturePoint(ip->lastTraceUV(0), ip->lastTraceUV(1), fhRight, ip->my_type, &Hcalib));

			ipRight->u_stereo = ipRight->u;
			ipRight->v_stereo = ipRight->v;
			ipRight->idepth_min_stereo = ip->idepth_min = 0;
			ipRight->idepth_max_stereo = ip->idepth_max = NAN;
//        std::cout << "idx: " << ip->idxInImmaturePoints << "\t Left." << std::endl;
			ImmaturePointStatus phTraceLeftStatus = ipRight->traceStereo(fh, K, 0);

			float u_stereo_delta = abs(ip->u_stereo - ipRight->lastTraceUV(0));
			float depth = 1.0f / ip->idepth_stereo;

			if (phTraceLeftStatus == ImmaturePointStatus::IPS_GOOD && u_stereo_delta < 1 && depth > 0 &&
			    depth < 50 * baseline) //original u_stereo_delta 1 depth < 70
			{

				ip->idepth_min = ip->idepth_min_stereo;
				ip->idepth_max = ip->idepth_max_stereo;

//          if (rand() / (float) RAND_MAX > debugKeepPercentage) continue;
//          keypoints_left.emplace_back(ip->u, ip->v, 1);
//          keypoints_right.emplace_back(ip->lastTraceUV(0), ip->lastTraceUV(1), 1);
//          matches.emplace_back(keypoints_left.size() - 1, keypoints_right.size() - 1, 1.0f);

				//ip->mF->_status = Feature::ACTIVE_IDEPTH;
				//这句给深度有问题 ip->mF->idepth = ip->idepth_stereo;
			}

			delete ipRight;
		}
	}
}

/**
 * [FullSystem::stereoMatch description]
 * @param image       [description]
 * @param image_right [description]
 * @param id          [description]
 * @param idepthMap   [description]
 */
void FullSystem::stereoMatch( ImageAndExposure * image, ImageAndExposure * image_right, int id, cv::Mat & idepthMap)
{
	// =========================== add into allFrameHistory =========================

	FrameHessian* fh(new FrameHessian());
	FrameHessian* fhRight(new FrameHessian());
	FrameShell* shell = new FrameShell();
	shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
	shell->aff_g2l = AffLight(0, 0);
	shell->marginalizedAt = shell->id = allFrameHistory.size();
	shell->timestamp = image->timestamp;

	//一直递增的id
	shell->incoming_id = id; // id passed into DSO

	//当前帧的信息
	fh->shell = shell;
	fhRight->shell = shell;

	// =========================== make Images / derivatives etc. =========================
	fh->ab_exposure = image->exposure_time;

	//左图的梯度点
	fh->makeImages(image, &Hcalib);
	fhRight->ab_exposure = image_right->exposure_time;

	//右图的梯度点
	fhRight->makeImages(image_right, &Hcalib);

	//内参
	Mat33f K = Mat33f::Identity();
	K(0, 0) = Hcalib.fxl();
	K(1, 1) = Hcalib.fyl();
	K(0, 2) = Hcalib.cxl();
	K(1, 2) = Hcalib.cyl();

	//计数
	int counter = 0;

	//创建新的一帧中的点
	makeNewTraces(fh, fhRight, 0);

	//逆深度图
	unsigned  char * idepthMapPtr = idepthMap.data;

	std::vector<cv::KeyPoint> keypoints_left;
	std::vector<cv::KeyPoint> keypoints_right;
	std::vector<cv::DMatch> matches;

	//遍历每一个点
	for (ImmaturePoint* ph : fh->immaturePoints)
	{
		//坐标
		ph->u_stereo = ph->u;
		ph->v_stereo = ph->v;
		ph->idepth_min_stereo = ph->idepth_min = 0;
		ph->idepth_max_stereo = ph->idepth_max = NAN;

		//左图与右图进行双目匹配，获取静态深度
		ImmaturePointStatus phTraceRightStatus = ph->traceStereo(fhRight, K, 1);

		//判断当前点的深度值是否好
		if (phTraceRightStatus == ImmaturePointStatus::IPS_GOOD)
		{
			//获取右图中的这个点
			ImmaturePoint* phRight(new ImmaturePoint(ph->lastTraceUV(0), ph->lastTraceUV(1), fhRight, &Hcalib ));

			//获取右图中的像素坐标
			phRight->u_stereo = phRight->u;
			phRight->v_stereo = phRight->v;
			phRight->idepth_min_stereo = ph->idepth_min = 0;
			phRight->idepth_max_stereo = ph->idepth_max = NAN;

			//右图与左图进行双目匹配，获取静态深度
			ImmaturePointStatus  phTraceLeftStatus = phRight->traceStereo(fh, K, 0);

			//两张图中的ｕ坐标差
			float u_stereo_delta = abs(ph->u_stereo - phRight->lastTraceUV(0));

			//左图中这个点的深度
			float depth = 1.0f / ph->idepth_stereo;

			//判断这个点的状态，这个点左图ｕ坐标小于１，深度在０-70之间
			if (phTraceLeftStatus == ImmaturePointStatus::IPS_GOOD && u_stereo_delta < 1 && depth > 0 && depth < 50 * baseline)   //original u_stereo_delta 1 depth < 70
			{
				keypoints_left.emplace_back(ph->u, ph->v, 1);
				keypoints_right.emplace_back(ph->lastTraceUV(0), ph->lastTraceUV(1), 1);
				matches.emplace_back(keypoints_left.size() - 1, keypoints_right.size() - 1, 1.0f);

				//更新该点的最小和最大的深度
				ph->idepth_min = ph->idepth_min_stereo;
				ph->idepth_max = ph->idepth_max_stereo;

				//更新逆深度图
				*((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u * 3) = ph->idepth_stereo;
				*((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u * 3 + 1) = ph->idepth_min;
				*((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u * 3 + 2) = ph->idepth_max;

				counter++;
			}

			delete phRight;
		}
	}

//    std::sort(error.begin(), error.end());
//    std::cout << 0.25 <<" "<<error[error.size()*0.25].first<<" "<<
//              0.5 <<" "<<error[error.size()*0.5].first<<" "<<
//              0.75 <<" "<<error[error.size()*0.75].first<<" "<<
//              0.1 <<" "<<error.back().first << std::endl;

//    for(int i = 0; i < error.size(); i++)
//        std::cout << error[i].first << " " << error[i].second.first << " " << error[i].second.second << std::endl;

	LOG(INFO) << " frameID " << id << " got good matches " << counter << std::endl;

	cv::Mat matLeft(image->h, image->w, CV_32F, image->image);
	cv::Mat matRight(image_right->h, image_right->w, CV_32F, image_right->image);
	matLeft.convertTo(matLeft, CV_8UC3);
	matRight.convertTo(matRight, CV_8UC3);

	cv::Mat matMatches;
	cv::drawMatches(matLeft, keypoints_left, matRight, keypoints_right, matches, matMatches);
	cv::imshow("matches", matMatches);
	cv::waitKey(0);

	delete fh;
	delete fhRight;
	return;
}

// process nonkey frame to refine key frame idepth
/**
 * [FullSystem::traceNewCoarseNonKey description]
 * @param fh       [description]
 * @param fhRight [description]
 * 非关键帧时,将前面的关键帧的点都投影到当前帧,进行traceOn,进行点逆深度滤波,并使用两次traceStereo,跟新这个点的最小和最大逆深度
 */
void FullSystem::traceNewCoarseNonKey(FrameHessian* fh, FrameHessian* fhRight)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	//新的逆深度最大和最小值
	// new idepth after refinement
	float idepth_min_update = 0;
	float idepth_max_update = 0;

	//内参
	Mat33f K = Mat33f::Identity();
	K(0, 0) = Hcalib.fxl();
	K(1, 1) = Hcalib.fyl();
	K(0, 2) = Hcalib.cxl();
	K(1, 2) = Hcalib.cyl();

	//内参的逆
	Mat33f Ki = K.inverse();

	//遍历每一个关键帧
	for (FrameHessian* host : frameHessians)        // go through all active frames
	{
		//个数
		// number++;
		int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

		//参考帧到当前帧位姿,将其投影到当前帧
		// trans from reference keyframe to newest frame
		SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
		// KRK-1
		Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
		// KRi
		Mat33f KRi = K * hostToNew.rotationMatrix().inverse().cast<float>();
		// Kt
		Vec3f Kt = K * hostToNew.translation().cast<float>();
		// t
		Vec3f t = hostToNew.translation().cast<float>();

		//aff
		Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

		//遍历该帧中每一个点
		for (ImmaturePoint* ph : host->immaturePoints)
		{
			//进行点的跟踪
			// do temperol stereo match
			ImmaturePointStatus phTrackStatus = ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

			//如果是好的点
			if (phTrackStatus == ImmaturePointStatus::IPS_GOOD)
			{
				//新建一个点
				ImmaturePoint* phNonKey = new ImmaturePoint(ph->lastTraceUV(0), ph->lastTraceUV(1), fh, &Hcalib);

				// project onto newest frame
				//重投影到新一帧，根据之前的最小逆深度
				Vec3f ptpMin = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_min) + Kt;
				//重投影后的最小逆深度值
				float idepth_min_project = 1.0f / ptpMin[2];
				//重投影到新一帧，根据之前的最大逆深度
				Vec3f ptpMax = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_max) + Kt;
				//重投影后的最大逆深度值
				float idepth_max_project = 1.0f / ptpMax[2];

				//重新设置该点的最小和最大逆深度
				phNonKey->idepth_min = idepth_min_project;
				phNonKey->idepth_max = idepth_max_project;
				//设置改点u,v坐标
				phNonKey->u_stereo = phNonKey->u;
				phNonKey->v_stereo = phNonKey->v;
				//重新设置该点的最小和最大逆深度
				phNonKey->idepth_min_stereo = phNonKey->idepth_min;
				phNonKey->idepth_max_stereo = phNonKey->idepth_max;

				// do static stereo match from left image to right
				// 进行双目静态的逆深度滤波，左目到右目
				ImmaturePointStatus phNonKeyStereoStatus = phNonKey->traceStereo(fhRight, K, 1);

				//静态状态是好的
				if (phNonKeyStereoStatus == ImmaturePointStatus::IPS_GOOD)
				{
					//右边点
					ImmaturePoint* phNonKeyRight = new ImmaturePoint(phNonKey->lastTraceUV(0), phNonKey->lastTraceUV(1), fhRight, &Hcalib );

					phNonKeyRight->u_stereo = phNonKeyRight->u;
					phNonKeyRight->v_stereo = phNonKeyRight->v;
					phNonKeyRight->idepth_min_stereo = phNonKey->idepth_min;
					phNonKeyRight->idepth_max_stereo = phNonKey->idepth_max;

					// do static stereo match from right image to left
					// 进行双目静态的逆深度滤波，右目到左目
					ImmaturePointStatus  phNonKeyRightStereoStatus = phNonKeyRight->traceStereo(fh, K, 0);

					// change of u after two different stereo match
					// 两次得到u坐标的绝对差值
					float u_stereo_delta = abs(phNonKey->u_stereo - phNonKeyRight->lastTraceUV(0));
					//
					float disparity = phNonKey->u_stereo - phNonKey->lastTraceUV[0];

					// free to debug the threshold
					// 差值过大，或者视差小于10，则out点
					if (u_stereo_delta > 1 && disparity < 10)
					{
						//跟新这个点的状态
						ph->lastTraceStatus = ImmaturePointStatus :: IPS_OUTLIER;

						delete phNonKey;
						delete phNonKeyRight;

						continue;
					}
					else
					{
						//重投影该点，更新最小和最大逆深度
						//将这个点在最大和最小逆深度处投影回关键帧,更新这个点的最小和最大逆深度
						// project back
						Vec3f pinverse_min = KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) / phNonKey->idepth_min_stereo - t);
						idepth_min_update = 1.0f / pinverse_min(2);

						Vec3f pinverse_max = KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) / phNonKey->idepth_max_stereo - t);
						idepth_max_update = 1.0f / pinverse_max(2);

						//更新这个点的最小和最大逆深度
						ph->idepth_min = idepth_min_update;
						ph->idepth_max = idepth_max_update;

						delete phNonKey;
						delete phNonKeyRight;
					}
				}
				else
				{
					delete phNonKey;
					continue;
				}
			}

			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
			trace_total++;
		}
	}
}


//process keyframe
/**
 * [FullSystem::traceNewCoarseKey description]
 * @param fh       [description]
 * @param fhRight [description]
 * 遍历每一个关键帧的中的点,将关键帧的点投影到当前帧,使用traceOn,更新逆深度
 */
void FullSystem::traceNewCoarseKey(FrameHessian* fh, FrameHessian* fhRight)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

	//内参
	Mat33f K = Mat33f::Identity();
	K(0, 0) = Hcalib.fxl();
	K(1, 1) = Hcalib.fyl();
	K(0, 2) = Hcalib.cxl();
	K(1, 2) = Hcalib.cyl();

	//遍历每一个关键帧的中的点,将关键帧的点投影到当前帧,进行traceOn,更新逆深度
	for (FrameHessian* host : frameHessians)		// go through all active frames
	{
		// trans from reference key frame to the newest one
		SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
		//KRK-1
		Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
		//Kt
		Vec3f Kt = K * hostToNew.translation().cast<float>();

		Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

		for (ImmaturePoint* ph : host->immaturePoints)
		{
			//点的深度滤波
			ImmaturePointStatus phTrackStatus = ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );

			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
			trace_total++;
		}
	}
}

/**
 * [FullSystem::activatePointsMT_Reductor description]
 * @param optimized  [description]	优化后的点
 * @param toOptimize [description]	优化前的点
 * @param min        [description]	最小的个数
 * @param max        [description]	最大的个数
 * @param stats      [description]	当前状态
 * @param tid        [description]
 * 从选出的ImmaturePoint点中生成实际的PointHessian点，生成深度
 */
void FullSystem::activatePointsMT_Reductor(
  std::vector<PointHessian*>* optimized,
  std::vector<ImmaturePoint*>* toOptimize,
  int min, int max, Vec10 * stats, int tid)
{
	ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];

	//优化每一个点从toOptimize中生成optimized
	for (int k = min; k < max; k++)
	{
		(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k], 1, tr);
	}
	delete[] tr;
}

/**
 * [FullSystem::activatePointsMT description]
 * 遍历窗口中的每一个关键帧的每一个点，判断这个点的状态并且将这个点与每一个关键帧进行逆深度残差更新，更新该点的逆深度
 * 并在ef中插入该点，加入该点与每一个关键帧的残差
 * 为每个关键帧从其ImmaturePoint中生成PointHessian点
 */
void FullSystem::activatePointsMT()
{
	//点个数
	if (ef->nPoints < setting_desiredPointDensity * 0.66) //setting_desiredPointDensity 是2000
		currentMinActDist -= 0.8;  //original 0.8
	if (ef->nPoints < setting_desiredPointDensity * 0.8)
		currentMinActDist -= 0.5;  //original 0.5
	else if (ef->nPoints < setting_desiredPointDensity * 0.9)
		currentMinActDist -= 0.2;  //original 0.2
	else if (ef->nPoints < setting_desiredPointDensity)
		currentMinActDist -= 0.1;  //original 0.1

	if (ef->nPoints > setting_desiredPointDensity * 1.5)
		currentMinActDist += 0.8;
	if (ef->nPoints > setting_desiredPointDensity * 1.3)
		currentMinActDist += 0.5;
	if (ef->nPoints > setting_desiredPointDensity * 1.15)
		currentMinActDist += 0.2;
	if (ef->nPoints > setting_desiredPointDensity)
		currentMinActDist += 0.1;

	//currentMinActDist在０－４之间
	if (currentMinActDist < 0) currentMinActDist = 0;
	if (currentMinActDist > 4) currentMinActDist = 4;

	if (!setting_debugout_runquiet)
		printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
		       currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);

	//最新的一关键帧
	FrameHessian* newestHs = frameHessians.back();

	// make dist map.
	//设置内参
	coarseDistanceMap->makeK(&Hcalib);
	//创建距离图
	coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

	//coarseTracker->debugPlotDistMap("distMap");

	//待优化的每一个点
	std::vector<ImmaturePoint*> toOptimize;
	//最大20000个点
	toOptimize.reserve(20000);

	//遍历窗口中的每一帧，选择待优化的点toOptimize
	for (FrameHessian* host : frameHessians)		// go through all active frames
	{
		if (host == newestHs) continue;

		// LOG(INFO) << "activatePointsMT before: " << host->frameID << " " << host->pointHessians.size() << " " << host->pointHessiansOut.size() << " " << host->pointHessiansMarginalized.size()
		//           << " " << host->immaturePoints.size() << " " << host->_features.size() << std::endl;

		//最新关键帧与主导帧的相对坐标
		SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;

		//K*R*K'
		Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
		//K*t
		Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

		// for all immaturePoints in frameHessian
		//遍历每一个主导中的点
		for (unsigned int i = 0; i < host->immaturePoints.size(); i += 1)
		{
			//点
			ImmaturePoint* ph = host->immaturePoints[i];
			//点id
			ph->idxInImmaturePoints = i;

			// delete points that have never been traced successfully, or that are outlier on the last trace.
			//判断点的状态，删除未成功跟踪的点，或删除最后一个跟踪点上的离群点
			//idepth_max初始为NAN,idepth_MIN初始为0
			if (!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER)
			{
//				immature_invalid_deleted++;
				// remove point.
				//删除该点
				delete ph;
				host->immaturePoints[i] = 0;
				continue;
			}

			// can activate only if this is true.
			// 这个点是否能激活,outlier和IPS_UNINITIALIZED不进行激活
			// ImmaturePoint初始为IPS_UNINITIALIZED
			// quality>3
			// lastTracePixelInterval<8
			// idepth_max和idepth_min有值了且和大于0
			bool canActivate = (ph->lastTraceStatus == IPS_GOOD
			                    || ph->lastTraceStatus == IPS_SKIPPED
			                    || ph->lastTraceStatus == IPS_BADCONDITION
			                    || ph->lastTraceStatus == IPS_OOB )
			                   && ph->lastTracePixelInterval < 8
			                   && ph->quality > setting_minTraceQuality
			                   && (ph->idepth_max + ph->idepth_min) > 0;

			// if I cannot activate the point, skip it. Maybe also delete it.
			//若不能激活，则删除这个点
			if (!canActivate)
			{
				// if point will be out afterwards, delete it instead.
				//若该点的主导帧已经被边缘化或者当前跟踪状态为OOB
				if (ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
				{
//					immature_notReady_deleted++;
					//删除该点
					delete ph;
					//该点为空
					host->immaturePoints[i] = 0;
				}
//				immature_notReady_skipped++;
				continue;
			}

			// see if we need to activate point due to distance map.
			//重投影该点
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * (0.5f * (ph->idepth_max + ph->idepth_min));
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;

			//该点在当前帧的坐标是否在画面内
			if ((u > 0 && v > 0 && u < wG[1] && v < hG[1]))
			{
				//该点的距离
				float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u + wG[1] * v] + (ptp[0] - floorf((float)(ptp[0])));

				//若距离大于currentMinActDist * ph->my_type
				if (dist >= currentMinActDist * ph->my_type)
				{
					//则距离图中插入该坐标
					coarseDistanceMap->addIntoDistFinal(u, v);
					//待优化点插入该点
					toOptimize.push_back(ph);
				}
			}
			else
			{
				//删除该点
				delete ph;
				host->immaturePoints[i] = 0; //删除点的操作
			}
		}
	}

//	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
//			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

	//优化后的每一个点
	std::vector<PointHessian*> optimized;
	optimized.resize(toOptimize.size());

	// std::cout<<"toOptimize: "<<toOptimize.size()<<std::endl;
	//多线程优化每一个点的逆深度
	//多线程生成PointHessian点,为每个关键帧帧生成PointHessian点
	if (multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);
	else
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);

	//遍历每一个优化前的点
	for (unsigned k = 0; k < toOptimize.size(); k++)
	{
		//该点优化后的
		PointHessian* newpoint = optimized[k];

		//之前的
		ImmaturePoint* ph = toOptimize[k];

		//新的点好的
		if (newpoint != 0 && newpoint != nullptr)//(PointHessian*)((long)(-1)))
		{
			//新的点
			newpoint->host->immaturePoints[ph->idxInImmaturePoints] = 0;

			if (ph->feaMode)
			{
				ph->mF->mPH = newpoint;
				newpoint->mF = ph->mF;
				newpoint->mF->_status = Feature::ACTIVE_PH;

				newpoint->feaMode = 1;
			}

			//该点的主导帧的点插入该点
			newpoint->host->pointHessians.push_back(newpoint);

			//误差函数中加入该点
			ef->insertPoint(newpoint);

			//遍历每一个点与帧的残差，ef中插入该残差
			for (PointFrameResidual* r : newpoint->residuals)
			{
				if (r->staticStereo) { //- static stereo residual
					ef->insertStaticResidual(r);
				}
				else
					ef->insertResidual(r);
			}
			assert(newpoint->efPoint != 0);
			delete ph;
		}
		else if (newpoint == nullptr //(PointHessian*)((long)(-1))
		         || ph->lastTraceStatus == IPS_OOB)
		{
			//删除该点
			delete ph;
			ph->host->immaturePoints[ph->idxInImmaturePoints] = 0;
		}
		else
		{
			assert(newpoint == 0 || newpoint == nullptr);//(PointHessian*)((long)(-1)));
		}
	}

	//遍历每一个主导帧
	for (FrameHessian* host : frameHessians)
	{

		// LOG(INFO) << "activatePointsMT after1: " << host->frameID << " " << host->pointHessians.size() << " " << host->pointHessiansOut.size() << " " << host->pointHessiansMarginalized.size()
		//           << " " << host->immaturePoints.size() << " " << host->_features.size() << std::endl;
		//遍历每一个主导帧的点
		for (int i = 0; i < (int)host->immaturePoints.size(); i++)
		{
			//若该点未被优化
			if (host->immaturePoints[i] == 0)
			{
				//则删除这个点
				host->immaturePoints[i] = host->immaturePoints.back();
				host->immaturePoints.pop_back();
				i--;
			}
		}
		// LOG(INFO) << "activatePointsMT after2: " << host->frameID << " " << host->pointHessians.size() << " " << host->pointHessiansOut.size() << " " << host->pointHessiansMarginalized.size()
		//           << " " << host->immaturePoints.size() << " " << host->_features.size() << std::endl;
	}
}

/**
 * [FullSystem::activatePointsOldFirst description]
 */
void FullSystem::activatePointsOldFirst()
{
	assert(false);
}

/**
 * [FullSystem::flagPointsForRemoval description]
 * 优化后，删除点
 * host->pointHessiansOut.push_back(ph);
 * efPoint->stateFlag = EFPointStatus::PS_DROP或PS_MARGINALIZE
 * host->pointHessians[i] = 0;pointHessians删除
 */
void FullSystem::flagPointsForRemoval()
{
	assert(EFIndicesValid);

	//保持的帧和边缘化的帧
	std::vector<FrameHessian*> fhsToKeepPoints;
	std::vector<FrameHessian*> fhsToMargPoints;

	//if(setting_margPointVisWindow>0)
	{
		for (int i = ((int)frameHessians.size()) - 1; i >= 0 && i >= ((int)frameHessians.size()); i--)
			if (!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);

		for (int i = 0; i < (int)frameHessians.size(); i++)
			if (frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
	}

	//ef->setAdjointsF();
	//ef->setDeltaF(&Hcalib);
	int flag_oob = 0, flag_in = 0, flag_inin = 0, flag_nores = 0;

	//遍历每一个关键帧
	for (FrameHessian* host : frameHessians)		// go through all active frames
	{
		// LOG(INFO) << "flagPointsForRemoval before: " << host->frameID << " " << host->pointHessians.size() << " " << host->pointHessiansOut.size() << " " << host->pointHessiansMarginalized.size()
		//           << " " << host->immaturePoints.size() << " " << host->_features.size() << std::endl;
		//遍历每一个点
		for (unsigned int i = 0; i < host->pointHessians.size(); i++)
		{
			//这个点
			PointHessian* ph = host->pointHessians[i];
			if (ph == 0) continue;

			//这个点逆深度＝＝０，则这个点边缘化　　插入pointHessiansOut
			//该点的状态＝PS_DROP，
			if (ph->idepth_scaled < 0 || ph->residuals.size() == 0)
			{
				host->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				ph->setPointStatus(PointHessian::OUTLIER);
				if (ph->feaMode)
					ph->mF->_status = Feature::OUTLIER;

				host->pointHessians[i] = 0;
				flag_nores++;
			}
			//主导帧被边缘化，或者该点被观察到的帧数够小
			else if (ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization)
			{
				flag_oob++;
				if (ph->isInlierNew())
				{
					flag_in++;
					int ngoodRes = 0;
					//遍历该点的每个与帧的残差
					for (PointFrameResidual* r : ph->residuals)
					{
						r->resetOOB();
						//线性化残差,即重新计算点到目标帧的坐标,并且计算这个点到目标帧的雅克比和残差

						if (r->staticStereo)
							r->linearizeStatic(&Hcalib);
						else
							r->linearize(&Hcalib);

						r->efResidual->isLinearized = false;
						//更新残差
						r->applyRes(true);
						if (r->efResidual->isActive())
						{
							r->efResidual->fixLinearizationF(ef);
							ngoodRes++;
						}
					}
					//若该点的逆深度Hessian超出边缘阈值，则加入pointHessiansMarginalized，该点为边缘化点
					if (ph->idepth_hessian > setting_minIdepthH_marg)
					{
						flag_inin++;
						ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
						host->pointHessiansMarginalized.push_back(ph);
					}
					//否则该点为out点
					else
					{
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
						host->pointHessiansOut.push_back(ph);
					}

				}
				else 	//该点为out点
				{
					//该点设置为out
					host->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
					//printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
				}

				host->pointHessians[i] = 0;
			}
		}

		//删除每一个要移除的点
		for (int i = 0; i < (int)host->pointHessians.size(); i++)
		{
			if (host->pointHessians[i] == 0)
			{
				host->pointHessians[i] = host->pointHessians.back();
				host->pointHessians.pop_back();
				i--;
			}
		}
		// LOG(INFO) << "flagPointsForRemoval after: " << host->frameID << " " << host->pointHessians.size() << " " << host->pointHessiansOut.size() << " " << host->pointHessiansMarginalized.size()
		//           << " " << host->immaturePoints.size() << " " << host->_features.size() << std::endl;
	}

	LOG(INFO) << "Flag: nores: " << flag_nores << ", oob: " << flag_oob << ", marged: " << flag_inin << endl;
}

/**
 * [FullSystem::addActiveFrame description]
 * @param image       [description]
 * @param image_right [description]
 * @param id          [description]
 */
void FullSystem::addActiveFrame( ImageAndExposure * image, ImageAndExposure * image_right, int id )
{
	if (isLost) return;
	//跟踪线程中互斥锁
	boost::unique_lock<boost::mutex> lock(trackMutex);

	// =========================== add into allFrameHistory =========================
	// std::cout<<std::endl;
	LOG(INFO) << std::endl;
	LOG(INFO) << "addActiveFrame: " << id << " " << allFrameHistory.size() << std::endl;
	//新建一个帧Hessian类
	FrameHessian* fh(new FrameHessian());
	//新建一个帧的位姿信息
	FrameHessian* fhRight(new FrameHessian());

	FrameShell* shell = new FrameShell();
	//相机坐标系到世界坐标系的变换矩阵，单位矩阵
	shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
	//参数为a和b，0,0
	shell->aff_g2l = AffLight(0, 0);
	//帧id为当前历史帧的数量
	shell->marginalizedAt = shell->id = allFrameHistory.size();
	//时间
	shell->timestamp = image->timestamp;
	shell->incoming_id = id; // id passed into DSO
	//设FrameHessian的位姿信息
	fh->shell = shell;
	fhRight->shell = shell;


	fh->rightFrame = fhRight;
	fhRight->leftFrame = fh;

	// =========================== make Images / derivatives etc. =========================
	//曝光时间
	fh->ab_exposure = image->exposure_time;
	//得到当前帧的每一层的灰度图像和xy方向梯度值和xy梯度平方和，用于跟踪和初始化
	fh->makeImages(image, &Hcalib);

	//曝光时间
	fhRight->ab_exposure = image_right->exposure_time;
	//得到当前帧的每一层的灰度图像和xy方向梯度值和xy梯度平方和，用于跟踪和初始化
	fhRight->makeImages(image_right, &Hcalib);

	makeNewTraces(fh, fhRight, 0);
	fh->ComputeBoW(_vocab);

	// makeCurrentDepth(fh, fhRight);

	cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
	K.at<float>(0, 0) = Hcalib.fxl();
	K.at<float>(1, 1) = Hcalib.fyl();
	K.at<float>(0, 2) = Hcalib.cxl();
	K.at<float>(1, 2) = Hcalib.cyl();

	//将当前帧增加到历史记录中
	allFrameHistory.push_back(shell);

	if (!initialized)
	{
		// use initializer!
		if (coarseInitializer->frameID < 0)	// first frame set. fh is kept by coarseInitializer.
		{
			//设置初始的双目
			coarseInitializer->setFirstStereo(&Hcalib, fh, fhRight);
			//初始化成功
			initialized = true;
		}
		return;
	}
	else	// do front-end operation.
	{
		bool usePnP = false;
		SE3 initT1, initT;
		int cntInliers = 0;
		std::vector<bool> mvbOutlier;
		if (frameHessians.size() > 3)
		{
			TicToc t;
			boost::timer bt;
			std::vector<cv::DMatch> matches, goofMatches;
			FrameHessian* pKF = frameHessians[frameHessians.size() - 1];
			int nmatches = matcher->SearchByBoW(fh, pKF, matches);
			matcher->checkUVDistance(fh, pKF, matches, goofMatches);
			LOG(INFO) << "matche time size: " << t.toc() << "ms " << goofMatches.size() << std::endl;
			//matcher->showMatch(fh,pKF,goofMatches);

			vector<cv::Point3f> p3d;
			vector<cv::Point2f> p2d;
			vector<int> matchIdx;
			for (size_t k = 0; k < goofMatches.size(); k++)
			{
				auto &m = goofMatches[k];

				Feature* featKF = pKF->_features[m.trainIdx];
				Feature* featCurrent = fh->_features[m.queryIdx];

				//if (featKF->mImP && featKF->mImP->lastTraceStatus == ImmaturePointStatus::IPS_GOOD)
				//{
				if (featKF->_status == Feature::ACTIVE_IDEPTH)
				{
					// there should be a 3d point
					// ImmaturePoint* pt = featKF->mImP;
					Vec3 pose;
					featKF->ComputePos(pose);
					LOG(INFO) << "pKF pt3d: " << " " << pose[0] << " " << pose[1] << " " << pose[2] << std::endl;
					cv::Point3f pt3d(pose[0], pose[1], pose[2]);
					p3d.push_back(pt3d);
					LOG(INFO) << "mpCurrentKF p2d: " << " " << featCurrent->_pixel[0] << " " << featCurrent->_pixel[1] << std::endl;
					p2d.push_back(cv::Point2f(featCurrent->_pixel[0], featCurrent->_pixel[1]));
					matchIdx.push_back(k);
				}
			}

			if (pnpCv(initT1, p2d, p3d, K, cntInliers, mvbOutlier, initT))
			{
				if (checkEstimatedPose(cntInliers, initT))
					usePnP = true;
			}
		}

		// =========================== SWAP tracking reference?. =========================
		//如果当前关键帧的参考帧ID大于当前跟踪的参考帧ID
		if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID)
		{
			boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
			//交换当前跟踪和和关键帧跟踪，始终保持跟踪ID大于关键帧跟踪个ID?
			CoarseTracker* tmp = coarseTracker;
			coarseTracker = coarseTracker_forNewKF;
			coarseTracker_forNewKF = tmp;
		}

		//进行跟踪
		Vec4 tres = trackNewCoarse(fh, fhRight, initT, usePnP);
		if (!std::isfinite((double)tres[0]) || !std::isfinite((double)tres[1]) || !std::isfinite((double)tres[2]) || !std::isfinite((double)tres[3]))
		{
			printf("Initial Tracking failed: LOST!\n");
			isLost = true;
			return;
		}

		//是否需要加入关键帧
		bool needToMakeKF = false;

		//这里setting_keyframesPerSecond=0，所以不会跳进去
		//这个使用两帧的间隔时间判断，即两帧之间时间大于0.95/setting_keyframesPerSecond，则设为关键帧
		if (setting_keyframesPerSecond > 0)
		{
			needToMakeKF = allFrameHistory.size() == 1 ||
			               (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f / setting_keyframesPerSecond;
		}
		else
		{
			//当前帧和参考帧的a和b的变换
			//得到两帧间的曝光变化
			//refToFh[0]=a＝e^(aj-ai)*tj*ti^(-1),两帧间的光度曝光变化
			//refToFh[1]=b = 当前帧的b - refToFh[0]*当前帧的b
			//参考帧的曝光:两帧间的光度曝光变化
			//当前帧的曝光时间:fh->ab_exposure
			Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
			               coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

			//判断是否是
			// BRIGHTNESS CHECK
			//亮度判断，是否将当前帧作为关键帧
			//如果历史只有一帧，则该帧加入关键帧
			/*或者
			* 初始setting_kfGlobalWeight=1，
			* setting_maxShiftWeightT＝0.04*(640+480)
			* setting_maxShiftWeightR＝0.0*(640+480)
			* setting_maxShiftWeightRT＝0.02*(640+480)
			* setting_maxAffineWeight=2
			* 即论文中的公示 wf*f + wft*ft + wa*a > Tkf,
			* 所以tres[1]为两帧间像素点重投影的位移偏差，tres[2]为两帧间的旋转偏差，tres[3]为两帧间的旋转和位移偏差。
			* 这里偏差只变换这些位移，旋转，和变换矩阵后每个像素点的差值。
			* refToFh[a]＝e^(aj-ai)*tj*ti^(-1),两帧间的光度曝光变化
			*/
			//当前帧的tres[0]大于第一帧的均方根误差
			float delta = setting_kfGlobalWeight * setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0] + hG[0]) +
			              setting_kfGlobalWeight * setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0] + hG[0]) +
			              setting_kfGlobalWeight * setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0] + hG[0]) +
			              setting_kfGlobalWeight * setting_maxAffineWeight * fabs(logf((float)refToFh[0]));

			// BRIGHTNESS CHECK
			// 判断是否是关键帧
			// 第一帧或者delta够大，或者误差残差大于了第一次的两倍
			needToMakeKF = allFrameHistory.size() == 1 || delta > 1 || 2 * coarseTracker->firstCoarseRMSE < tres[0];
		}

		//显示位姿
		for (IOWrap::Output3DWrapper* ow : outputWrapper)
			ow->publishCamPose(fh->shell, &Hcalib);

		lock.unlock();

		//传递到后端，是否加入关键帧判断是否后端优化
		deliverTrackedFrame(fh, fhRight, needToMakeKF);
		return;
	}
}

/**
 * [FullSystem::deliverTrackedFrame description]
 * @param fh       [description]
 * @param fhRight [description]
 * @param needKF   [description]
 */
void FullSystem::deliverTrackedFrame(FrameHessian* fh, FrameHessian* fhRight, bool needKF)
{
	if (linearizeOperation)
	{
		//这里goStepByStep＝false，即是否一步一步显示
		//并且上一参考帧ID是不等于当前参考帧时，即参考帧有变化的时候，才显示
		if (goStepByStep && lastRefStopID != coarseTracker->refFrameID)
		{
			MinimalImageF3 img(wG[0], hG[0], fh->dI);
			//显示当前图像dI
			IOWrap::displayImage("frameToTrack", &img);
			while (true)
			{
				char k = IOWrap::waitKey(0);
				if (k == ' ') break;
				handleKey( k );
			}
			//更新参考帧id
			lastRefStopID = coarseTracker->refFrameID;
		}
		else handleKey( IOWrap::waitKey(1) );

		//加入关键帧
		if (needKF) makeKeyFrame(fh, fhRight);
		//不加入关键帧
		else makeNonKeyFrame(fh, fhRight);

	}
	//如果不使用linearizeOperation
	else
	{
		//上锁trackMapSyncMutex
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
		//mappingLoop中图像插入序列插入
		unmappedTrackedFrames.push_back(fh);
		unmappedTrackedFrames_right.push_back(fhRight);
		//是否是关键帧，设置最新的关键帧id
		if (needKF)
			needNewKFAfter = fh->shell->trackingRef->id;

		//通知处在等待该对象的线程的方法
		//notify_all唤醒所有正在等待该对象的线程,这里唤醒mappingLoop
		trackedFrameSignal.notify_all();

		//当跟踪器和新的跟踪器的的参考帧为-1时,即跟踪器和新的跟踪器没有设置时，则建图信号上锁
		//当前线程进行上锁,
		while (coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
		{
			mappedFrameSignal.wait(lock);
		}

		//解锁trackMapSyncMutex
		lock.unlock();
	}
}

/**
 * [FullSystem::mappingLoop description]
 * 后端优化
 * 若linearizeOperation=true，则这里不进行，
 * 若linearizeOperation=false,unmappedTrackedFrame才有值
 */
void FullSystem::mappingLoop()
{
	//非同步互斥锁,上锁trackMapSyncMutex
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

	while (runMapping)
	{
		//若unmappedTrackedFrames为０
		while (unmappedTrackedFrames.size() == 0)
		{
			//当前被锁住,.当前线程一直阻塞,等待deliverTrackedFrame中进行释放
			trackedFrameSignal.wait(lock);
			if (!runMapping) return;
		}

		//则此时unmappedTrackedFrames大小肯定大于0
		//获取最前面的一帧
		FrameHessian* fh = unmappedTrackedFrames.front();
		//弹出
		unmappedTrackedFrames.pop_front();

		//获取最前面的右图像的帧
		FrameHessian* fhRight = unmappedTrackedFrames_right.front();
		//弹出
		unmappedTrackedFrames_right.pop_front();

		// guaranteed to make a KF for the very first two tracked frames.
		// 保证为前两个跟踪帧制作一个KF。小于一帧的时候
		if (allKeyFramesHistory.size() <= 2)
		{
			//解锁,则此时deliverTrackedFrame线程和makeFrame线程同时进行
			lock.unlock();
			//插入关键帧
			makeKeyFrame(fh, fhRight);

			//上锁
			lock.lock();
			//唤醒deliverTrackedFrame的线程,此时已经插入了关键帧了,即跟新了跟踪器
			mappedFrameSignal.notify_all();
			continue;
		}

		//unmappedTrackedFrames>3的时，需要进行needToKetchupMapping
		//即意味着deliverTrackedFrame的线程运行了5次,这个线程才运行了1次,则needToKetchupMapping=true
		if (unmappedTrackedFrames.size() > 3)
			needToKetchupMapping = true;

		//needToKetchupMapping还大于0,说明unmappedTrackedFrames>1,则次帧为非关键帧
		if (unmappedTrackedFrames.size() > 0) // if there are other frames to track, do that first.
		{
			//解锁,则此时deliverTrackedFrame线程和makeFrame线程同时进行
			lock.unlock();
			//插入非关键帧
			makeNonKeyFrame(fh, fhRight);
			//上锁
			lock.lock();

			//这个needToKetchupMapping==true,说明优化比较慢的时候
			if (needToKetchupMapping && unmappedTrackedFrames.size() > 0)
			{
				//弹出最前面的一帧
				FrameHessian* fh = unmappedTrackedFrames.front();
				unmappedTrackedFrames.pop_front();
				{
					//位姿锁
					boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
					assert(fh->shell->trackingRef != 0);

					//当前帧位姿,设置当前帧位姿
					fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
					fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
				}
				//删除当前帧
				delete fh;
				delete fhRight;
			}
		}
		else
		{
			//setting_realTimeMaxKF=false	为true实时最大帧数,即几乎每一帧都为关键帧插入,如果相机静止,则会有问题
			//若当前最新的关键帧id大于关键帧序列中最后的一帧
			if (setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)
			{
				lock.unlock();
				//插入关键帧
				makeKeyFrame(fh, fhRight);
				//插完了关键帧,不进行KetchupMapping
				needToKetchupMapping = false;
				lock.lock();
			}
			else
			{
				lock.unlock();
				//插入非关键帧
				makeNonKeyFrame(fh, fhRight);
				lock.lock();
			}
		}
		//唤醒deliverTrackedFrame的线程,此时已经插入了关键帧了,即跟新了跟踪器
		mappedFrameSignal.notify_all();
	}
	LOG(INFO) << "MAPPING FINISHED!" << std::endl;
}

/**
 * [FullSystem::blockUntilMappingIsFinished description]
 */
void FullSystem::blockUntilMappingIsFinished()
{
	//上锁trackMapSyncMutex
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

	//关闭mapping
	runMapping = false;

	//唤醒所有正在等待该对象的线程
	trackedFrameSignal.notify_all();
	//解锁trackMapSyncMutex
	lock.unlock();

	//mapping线程阻塞
	mappingThread.join();

	loopClosing->setFinish(true);
}

/**
 * [FullSystem::makeNonKeyFrame description]
 * @param fh       [description]
 * @param fhRight [description]
 * 不将当前帧作为关键帧
 */
void FullSystem::makeNonKeyFrame( FrameHessian* fh, FrameHessian* fhRight)
{
	// needs to be set by mapping thread. no lock required since we are in mapping thread.
	LOG(INFO) << "makeNonKeyFrame: " << fh->shell->id << std::endl;

	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		//根据参考帧到世界坐标系的变换和当前帧和参考帧之间的变换，计算当前帧到世界坐标系的变换
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		//设置当前帧的位姿和a和b
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
	}

	//跟踪这一帧中的每个点，对这个点的像素坐标和状态进行更新
	traceNewCoarseNonKey(fh, fhRight);

	for (unsigned int i = 0; i < fh-> _features.size(); i++)
	{
		delete fh->_features[i];
	}

	for (unsigned int i = 0; i < fhRight->_features.size(); i++)
	{
		delete fhRight->_features[i];
	}

	//删除当前帧
	delete fh;
	delete fhRight;
}

void FullSystem::makeCurrentDepth(FrameHessian* fh, FrameHessian* fhRight)
{
//内参
	Mat33f K = Mat33f::Identity();
	K(0, 0) = Hcalib.fxl();
	K(1, 1) = Hcalib.fyl();
	K(0, 2) = Hcalib.cxl();
	K(1, 2) = Hcalib.cyl();

	//计数
	int counter = 0;

	//遍历每一个点
	for (ImmaturePoint* ph : fh->immaturePoints)
	{
		//坐标
		ph->u_stereo = ph->u;
		ph->v_stereo = ph->v;
		ph->idepth_min_stereo = ph->idepth_min = 0;
		ph->idepth_max_stereo = ph->idepth_max = NAN;

		//左图与右图进行双目匹配，获取静态深度
		ImmaturePointStatus phTraceRightStatus = ph->traceStereo(fhRight, K, 1);

		//判断当前点的深度值是否好
		if (phTraceRightStatus == ImmaturePointStatus::IPS_GOOD)
		{
			//获取右图中的这个点
			ImmaturePoint* phRight(new ImmaturePoint(ph->lastTraceUV(0), ph->lastTraceUV(1), fhRight, &Hcalib ));

			//获取右图中的像素坐标
			phRight->u_stereo = phRight->u;
			phRight->v_stereo = phRight->v;
			phRight->idepth_min_stereo = ph->idepth_min = 0;
			phRight->idepth_max_stereo = ph->idepth_max = NAN;

			//右图与左图进行双目匹配，获取静态深度
			ImmaturePointStatus  phTraceLeftStatus = phRight->traceStereo(fh, K, 0);

			//两张图中的ｕ坐标差
			float u_stereo_delta = abs(ph->u_stereo - phRight->lastTraceUV(0));

			//左图中这个点的深度
			float depth = 1.0f / ph->idepth_stereo;

			//判断这个点的状态，这个点左图ｕ坐标小于１，深度在０-70之间
			if (phTraceLeftStatus == ImmaturePointStatus::IPS_GOOD && u_stereo_delta < 1 && depth > 0 && depth < 70)   //original u_stereo_delta 1 depth < 70
			{
				//更新该点的最小和最大的深度
				ph->idepth_min = ph->idepth_min_stereo;
				ph->idepth_max = ph->idepth_max_stereo;
				ph->lastTraceStatus = ImmaturePointStatus :: IPS_GOOD;
			}

			delete phRight;
		}
	}
}

/**
 * [FullSystem::makeKeyFrame description]
 * @param fh       [description]
 * @param fhRight [description]
 *
 * 1.先对每个点进行更新
 * 2.判断窗口中的关键帧，是否边缘化关键帧
 * 3.设置每一个关键帧之间为主导帧
 * 4.加入每一个关键帧中的点与其它关键帧的残差
 * 5.遍历窗口中的每一个关键帧的每一个点，判断这个点的状态并且将这个点与每一个关键帧进行逆深度残差更新，更新该点的逆深度
 * 	并在ef中插入该点，加入该点与每一个关键帧的残差,为最新帧的主导帧从其ImmaturePoint中生成PointHessian点
 * 6.优化，最大优化次数6次
 * 7.移除外点removeOutliers
 * 8.设置新的跟踪器coarseTracker_forNewKF
 * 9.删除点，并在ef中删除点和并跟新ef中的H和b
 */
void FullSystem::makeKeyFrame( FrameHessian* fh, FrameHessian* fhRight)
{
	LOG(INFO) << "frame " << fh->shell->id << " is marked as key frame" << endl;
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		//根据参考帧到世界坐标系的变换和当前帧和参考帧之间的变换，计算当前帧到世界坐标系的变换
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		//设置当前帧的位姿和a和b
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
		fh->shell->camToWorldOpti = fh->shell->camToWorld;
	}

	//将之前关键帧的点全部投影到当前帧,使用traceOn计算当之前关键帧的点的逆深度
	traceNewCoarseKey(fh, fhRight);

	boost::unique_lock<boost::mutex> lock(mapMutex);

	// =========================== Flag Frames to be Marginalized. =========================
	//是否边缘化该帧
	//1. 前帧的点个数过小，则该帧被边缘化或者该帧与最新的帧的光度变化较大，且剩下的帧数大于最小帧数
	//2. 帧数大于最大帧数，则移除与其它帧距离和最大的一帧
	flagFramesForMarginalization(fh);

	// =========================== add New Frame to Hessian Struct. =========================
	// 加入新帧信息到Hessian矩阵中
	//窗口中关键帧id，若是６个窗口，则一直是６
	fh->idx = frameHessians.size();

	//插入Hessian帧,做为关键帧
	frameHessians.push_back(fh);
	frameHessiansRight.push_back(fhRight);

	//关键帧id
	fh->frameID = allKeyFramesHistory.size();

	//插入关键帧
	allKeyFramesHistory.push_back(fh->shell);

	// needs to be set by mapping thread
	LOG(INFO) << "makeKeyFrame " << fh->shell->id << " " << fh->idx << " " << fh->frameID << std::endl;

	//误差能量函数插入该帧的Hessian
	ef->insertFrame(fh, &Hcalib);

	//设置每一个关键帧之间为主导帧
	setPrecalcValues();

	// =========================== add new residuals for old points =========================
	//对于每一个旧点增加残差
	int numFwdResAdde = 0;
	//遍历每一个关键帧很
	for (FrameHessian* fh1 : frameHessians)		// go through all active frames
	{
		if (fh1 == fh)
			continue;
		//遍历每一个点
		for (PointHessian* ph : fh1->pointHessians)
		{
			//当前帧与这个关键帧这个点的残差
			PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
			//设置类型
			r->setState(ResState::IN);
			//加入残差
			ph->residuals.push_back(r);
			ef->insertResidual(r);

			//更新这个点的残差
			ph->lastResiduals[1] = ph->lastResiduals[0];
			ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);

			//个数++
			numFwdResAdde += 1;
		}
	}

	//每一帧的误差阈值
	fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;


	// =========================== Activate Points (& flag for marginalization). =========================
	// 这里生成了逆深度
	// 遍历窗口中的每一个关键帧的每一个点，判断这个点的状态并且将这个点与每一个关键帧进行逆深度残差更新，更新该点的逆深度
	// 并在ef中插入该点，加入该点与每一个关键帧的残差
	// 为最新帧的主导帧从其ImmaturePoint中生成PointHessian点
	activatePointsMT();

	//重新设置ef中帧和点的Idx,因为新加了点和帧
	ef->makeIDX();

	// =========================== OPTIMIZE ALL =========================
	//每一帧的误差阈值
	fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;

	//优化
	float rmse = optimize(setting_maxOptIterations);

	LOG(INFO) << "rmse is " << rmse << std::endl;

	// =========================== Figure Out if INITIALIZATION FAILED =========================
	//判断初始化是否成功
	if (allKeyFramesHistory.size() <= 4)
	{
		if (allKeyFramesHistory.size() == 2 && rmse > 20 * benchmark_initializerSlackFactor)
		{
			LOG(INFO) << "I THINK INITIALIZATINO FAILED! Resetting." << std::endl;
			initFailed = true;
		}
		if (allKeyFramesHistory.size() == 3 && rmse > 13 * benchmark_initializerSlackFactor)
		{
			LOG(INFO) << "I THINK INITIALIZATINO FAILED! Resetting." << std::endl;
			initFailed = true;
		}
		if (allKeyFramesHistory.size() == 4 && rmse > 9 * benchmark_initializerSlackFactor)
		{
			LOG(INFO) << "I THINK INITIALIZATINO FAILED! Resetting." << std::endl;
			initFailed = true;
		}
	}

	if (isLost)
		return;

	// =========================== REMOVE OUTLIER =========================
	//移除外点，删除点和窗口中的帧之间都无残差，则加入pointHessiansOut，并从pointHessians，在ef中删除PS_DROP
	removeOutliers();
	{
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
		//设置新的跟踪器的内参
		coarseTracker_forNewKF->makeK(&Hcalib);
		//设置新的跟踪器的参考帧，并且使用双目静态匹配获取参考帧的点的逆深度
		coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians, fhRight, Hcalib);
		//coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians, Hcalib);

		//发布深度图
		coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
		coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
	}

	//	debugPlot("post Optimize");
	// =========================== (Activate-)Marginalize Points =========================
	//边缘化点，删除点
	flagPointsForRemoval();

	//在ef误差函数中移除被边缘化的点
	ef->dropPointsF();

	//获取零空间
	getNullspaces(
	  ef->lastNullspaces_pose,
	  ef->lastNullspaces_scale,
	  ef->lastNullspaces_affA,
	  ef->lastNullspaces_affB);

	//边缘化点后，更新ef误差函数中的Ｈessian和ｂ矩阵
	ef->marginalizePointsF();

	// =========================== add new Immature points & new residuals =========================
	//获取当前新的关键帧的点,fh->immaturePoints
	//makeNewTraces(fh, fhRight, 0);

	//- use right frame to initialize the depth of fh->immaturePoints
	stereoMatch(fh, fhRight);


	Frame* f(new Frame(fh));
	globalMap->addKeyFrame(f);

	//record the relative poses, note we are building a covisibility graph in fact
	//获取窗口中最大和最小的关键帧id
	auto minandmax = std::minmax_element(frameHessians.begin(), frameHessians.end(), CmpFrameHessianKFID());
	unsigned long minKFId = (*minandmax.first)->frameID;
	unsigned long maxKFId = (*minandmax.second)->frameID;

	//输出最大和最小的关键帧id
	LOG(INFO) << "min max KFId = " << minKFId << ", " << maxKFId << endl;

	//b遍历窗口中所有关键帧
	for (auto &fh : frameHessians)
	{
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
			for (auto &f2 : allKFs)
			{
				//在窗口中id范围内全部的关键帧,且不等于当前窗口中的关键帧,进行关联,获取这两个关键帧的相对位姿
				if (f2->frameID > minKFId && f2->frameID < maxKFId && f2->frameID != f1->frameID)
				{
					std::unique_lock<std::mutex> lock(fh->mMutexPoseRel);
					f1->mPoseRel[f2] = f1->camToWorld.inverse() * f2->camToWorld;
					f2->mPoseRel[f1] = f2->camToWorld.inverse() * f1->camToWorld;
				}
			}
		}
		else
		{
			cout << "Cannot find the Frame!" << endl;
		}
		delete tempF;
	}

	//发布关键帧和当前窗口中帧的关联
	for (IOWrap::Output3DWrapper* ow : outputWrapper)
	{
		ow->publishGraph(ef->connectivityMap);
		ow->publishKeyframes(frameHessians, false, &Hcalib);
	}

	for(auto &fh : frameHessians)
	{
		for(auto &ph : fh->pointHessians)
		{
			if(ph->feaMode)
			{
				ph->mF->_status = Feature::ACTIVE_IDEPTH;
				ph->mF->idepth=ph->idepth_scaled;
				ph->mF->idepth_hessian=ph->idepth_hessian;
				ph->mF->maxRelBaseline=ph->maxRelBaseline;
			}
		}
		for(auto &ph : fh->pointHessiansMarginalized)
		{
			if(ph->feaMode)
			{
				ph->mF->_status = Feature::ACTIVE_IDEPTH;
				ph->mF->idepth=ph->idepth_scaled;
				ph->mF->idepth_hessian=ph->idepth_hessian;
				ph->mF->maxRelBaseline=ph->maxRelBaseline;
			}
		}
	}


	// =========================== Marginalize Frames =========================
	//边缘化帧
	LOG(INFO) << "Marginalize frame" << std::endl;
	for (unsigned int i = 0; i < frameHessians.size(); i++)
	{
		//该帧需要边缘化
		if (frameHessians[i]->flaggedForMarginalization)
		{
			//边缘化这一帧
			marginalizeFrame(frameHessians[i]);
			i = 0;
		}
	}

	//闭环检测插入当前关键帧
	loopClosing->insertKeyFrame(f);

	LOG(INFO) << "delete right frame" << std::endl;

	//delete fhRight;

//	printLogLine();
//    printEigenValLine();
}

void FullSystem:: initializeFromInitializer(FrameHessian* newFrame)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	// add firstframe.
	FrameHessian* firstFrame = coarseInitializer->firstFrame;
	firstFrame->idx = frameHessians.size();
	frameHessians.push_back(firstFrame);
	frameHessiansRight.push_back(firstFrame->rightFrame);
	firstFrame->frameID = allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(firstFrame->shell);
	ef->insertFrame(firstFrame, &Hcalib);
	setPrecalcValues();

	//int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	//int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

	firstFrame->pointHessians.reserve(wG[0] * hG[0] * 0.2f);
	firstFrame->pointHessiansMarginalized.reserve(wG[0] * hG[0] * 0.2f);
	firstFrame->pointHessiansOut.reserve(wG[0] * hG[0] * 0.2f);


	float sumID = 1e-5, numID = 1e-5;
	for (int i = 0; i < coarseInitializer->numPoints[0]; i++) {
		sumID += coarseInitializer->points[0][i].iR;
		numID++;
	}
	float rescaleFactor = 1 / (sumID / numID);

	// randomly sub-select the points I need.
	float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

	if (!setting_debugout_runquiet)
		printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100 * keepPercentage,
		       (int) (setting_desiredPointDensity), coarseInitializer->numPoints[0]);

	for (int i = 0; i < coarseInitializer->numPoints[0]; i++)
	{
		if (rand() / (float) RAND_MAX > keepPercentage) continue;

		Pnt *point = coarseInitializer->points[0] + i;
		ImmaturePoint* pt(new ImmaturePoint(point->u + 0.5f, point->v + 0.5f, firstFrame, point->my_type, &Hcalib));

		if (!std::isfinite(pt->energyTH)) {
			delete pt;
			continue;
		}

		pt->idepth_max = pt->idepth_min = 1;
		PointHessian* ph(new PointHessian(pt, &Hcalib));
		delete pt;
		if (!std::isfinite(ph->energyTH)) {
			delete ph;
			continue;
		}

		ph->setIdepthScaled(point->iR * rescaleFactor);
		ph->setIdepthZero(ph->idepth);
		ph->hasDepthPrior = true;
		ph->setPointStatus(PointHessian::ACTIVE);


		firstFrame->pointHessians.push_back(ph);
		ef->insertPoint(ph);
	}

	SE3 firstToNew = coarseInitializer->thisToNext;
	firstToNew.translation() /= rescaleFactor;


	// really no lock required, as we are initializing.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		firstFrame->shell->camToWorld = SE3();
		firstFrame->shell->aff_g2l = AffLight(0, 0);
		firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(), firstFrame->shell->aff_g2l);
		firstFrame->shell->trackingRef = 0;
		firstFrame->shell->camToTrackingRef = SE3();

		newFrame->shell->camToWorld = firstToNew.inverse();
		newFrame->shell->aff_g2l = AffLight(0, 0);
		newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(), newFrame->shell->aff_g2l);
		newFrame->shell->trackingRef = firstFrame->shell;
		newFrame->shell->camToTrackingRef = firstToNew.inverse();

	}

	initialized = true;
	LOG(INFO) << "INITIALIZE FROM INITIALIZER (" << (int)firstFrame->pointHessians.size() << " pts)!" << std::endl;
}
// insert the first Frame into FrameHessians
/**
 * [FullSystem::initializeFromInitializerStereo description]
 * @param newFrame [description]
 */
void FullSystem::initializeFromInitializerStereo(FrameHessian* newFrame)
{
	//地图上锁
	boost::unique_lock<boost::mutex> lock(mapMutex);

	//设置内参
	Mat33f K = Mat33f::Identity();
	K(0, 0) = Hcalib.fxl();
	K(1, 1) = Hcalib.fyl();
	K(0, 2) = Hcalib.cxl();
	K(1, 2) = Hcalib.cyl();

	// add firstframe.
	// 第一帧
	FrameHessian* firstFrame = coarseInitializer->firstFrame;
	//关键帧id
	firstFrame->idx = frameHessians.size();
	//插入这一帧,
	frameHessians.push_back(firstFrame);
	//关键帧id
	firstFrame->frameID = allKeyFramesHistory.size();
	//插入这一帧的信息
	allKeyFramesHistory.push_back(firstFrame->shell);

	//插入当前关键帧
	Frame* f(new Frame(firstFrame));
	globalMap->addKeyFrame(f);

	//能量函数插入当前帧
	ef->insertFrame(firstFrame, &Hcalib);
	//设置每一帧的目标帧，这时候只有第一帧
	setPrecalcValues();

	//第一帧的右帧
	FrameHessian* firstFrameRight = coarseInitializer->firstRightFrame;
	// //
	frameHessiansRight.push_back(firstFrameRight);

	//设置第一帧的点Hessian矩阵
	firstFrame->pointHessians.reserve(wG[0]*hG[0] * 0.2f);
	firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0] * 0.2f);
	firstFrame->pointHessiansOut.reserve(wG[0]*hG[0] * 0.2f);

	float idepthStereo = 0;
	float sumID = 1e-5, numID = 1e-5;

	//遍历每一个点
	for (int i = 0; i < coarseInitializer->numPoints[0]; i++)
	{
		sumID += coarseInitializer->points[0][i].iR;
		numID++;
	}

	// randomly sub-select the points I need.
	// 随机采样
	float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

	LOG(INFO) << "Initialization: keep " << 100 * keepPercentage << "(need " << (int)(setting_desiredPointDensity)
	          << " have " << coarseInitializer->numPoints[0] << std::endl;

	// initialize first frame by idepth computed by static stereo matching
	// 遍历每一个点
	for (int i = 0; i < coarseInitializer->numPoints[0]; i++)
	{
		if (rand() / (float)RAND_MAX > keepPercentage) continue;

		Pnt* point = coarseInitializer->points[0] + i;

		//初始化一个点
		ImmaturePoint* pt(new ImmaturePoint(point->u + 0.5f, point->v + 0.5f, firstFrame, point->my_type, &Hcalib));

		//设置该点的坐标和最小和最大的逆深度
		pt->u_stereo = pt->u;
		pt->v_stereo = pt->v;
		pt->idepth_min_stereo = 0;
		pt->idepth_max_stereo = NAN;

		//静态双目跟踪，左图与右图进行匹配
		const ImmaturePointStatus ipTraceRightStatus = pt->traceStereo(firstFrameRight, K, 1);

		if (ipTraceRightStatus == ImmaturePointStatus::IPS_GOOD)
		{
			ImmaturePoint* ipRight(new ImmaturePoint(pt->lastTraceUV(0), pt->lastTraceUV(1), firstFrameRight,
			                       point->my_type, &Hcalib));

			ipRight->u_stereo = ipRight->u;
			ipRight->v_stereo = ipRight->v;
			ipRight->idepth_min_stereo = ipRight->idepth_min = 0;
			ipRight->idepth_max_stereo = ipRight->idepth_max = 0;

			const ImmaturePointStatus ipTraceLeftStatus = ipRight->traceStereo(firstFrame, K, 0);

			float u_stereo_delta = abs(pt->u_stereo - ipRight->lastTraceUV(0));
			float depth = 1.0f / pt->idepth_stereo;

			if (ipTraceLeftStatus == ImmaturePointStatus::IPS_GOOD && u_stereo_delta < 1 && depth > 0 && depth < 50 * baseline)
			{
				//设置点的最小和最大逆深度
				pt->idepth_min = pt->idepth_min_stereo;
				pt->idepth_max = pt->idepth_max_stereo;
				//idepthStereo = pt->idepth_stereo;

				// //判断该点的最小和最大的逆深度
				// if (!std::isfinite(pt->energyTH) || !std::isfinite(pt->idepth_min) || !std::isfinite(pt->idepth_max)
				//     || pt->idepth_min < 0 || pt->idepth_max < 0)
				// {
				// 	delete pt;
				// 	continue;
				// }

				//创建该点的Hessian矩阵
				PointHessian* ph(new PointHessian(pt, &Hcalib));

				delete pt;
				if (!std::isfinite(ph->energyTH))
				{
					delete ph;
					continue;
				}

				//插入该点
				ph->setIdepthScaled(idepthStereo);
				ph->setIdepthZero(idepthStereo);
				//是否有逆深度的初值，该点有初始的逆深度
				ph->hasDepthPrior = true;
				//设置点的状态，激活状态
				ph->setPointStatus(PointHessian::ACTIVE);

				if (ph->feaMode)
				{
					ph->mF->_status = Feature::ACTIVE_IDEPTH;
					ph->mF->idepth = idepthStereo;
				}

				//该帧插入该点
				firstFrame->pointHessians.push_back(ph);

				//ef插入该点
				ef->insertPoint(ph);
			}
		}

		pt->idepth_min = pt->idepth_min_stereo;
		pt->idepth_max = pt->idepth_max_stereo;
		idepthStereo = pt->idepth_stereo;

		//判断该点的最小和最大的逆深度
		if (!std::isfinite(pt->energyTH) || !std::isfinite(pt->idepth_min) || !std::isfinite(pt->idepth_max)
		    || pt->idepth_min < 0 || pt->idepth_max < 0)
		{
			delete pt;
			continue;
		}

		//创建该点的Hessian矩阵
		PointHessian* ph(new PointHessian(pt, &Hcalib));

		if (pt->feaMode)
		{
			pt->mF->mPH = ph;
			ph->mF = pt->mF;
			ph->mF->_status = Feature::ACTIVE_PH;
			ph->mF->idepth = idepthStereo;

			ph->feaMode = 1;
		}

		delete pt;
		if (!std::isfinite(ph->energyTH))
		{
			delete ph;
			continue;
		}

		//插入该点
		ph->setIdepthScaled(idepthStereo);
		ph->setIdepthZero(idepthStereo);
		//是否有逆深度的初值，该点有初始的逆深度
		ph->hasDepthPrior = true;
		//设置点的状态，激活状态
		ph->setPointStatus(PointHessian::ACTIVE);

		//该帧插入该点
		firstFrame->pointHessians.push_back(ph);

		//ef插入该点
		ef->insertPoint(ph);
	}

	//第一帧到最新一阵的位姿变换
	SE3 firstToNew = coarseInitializer->thisToNext;

	//设置这两帧的位姿
	// really no lock required, as we are initializing.
	{
		//设置第一帧
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		firstFrame->shell->camToWorld = SE3();
		firstFrame->shell->aff_g2l = AffLight(0, 0);
		firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(), firstFrame->shell->aff_g2l);
		firstFrame->shell->trackingRef = 0;
		firstFrame->shell->camToTrackingRef = SE3();

		//设置最新的一帧
		newFrame->shell->camToWorld = firstToNew.inverse();
		newFrame->shell->aff_g2l = AffLight(0, 0);
		newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(), newFrame->shell->aff_g2l);
		newFrame->shell->trackingRef = firstFrame->shell;
		newFrame->shell->camToTrackingRef = firstToNew.inverse();

	}

	//初始化成功
	initialized = true;
	LOG(INFO) << "INITIALIZE FROM INITIALIZER (" << (int)firstFrame->pointHessians.size() << " pts)!" << std::endl;
}

/**
 * [FullSystem::makeNewTraces description]
 * @param newFrame      [description]
 * @param newFrameRight [description]
 * @param gtDepth       [description]
 * 选取新关键帧的点,创建ImmaturePoint
 */
void FullSystem::makeNewTraces(FrameHessian* newFrame, FrameHessian* newFrameRight, float * gtDepth)
{
	boost::timer t;

	pixelSelector->allowFast = true;
	//int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	//筛选新的点，点的总数
	int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap, setting_desiredImmatureDensity);

	//设置新参考帧的点Hessian矩阵
	newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
	//fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
	newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);

	//遍历每一个点,selectionMap==0则不是。1则是
	for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++)
		for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++)
		{
			int i = x + y * wG[0];
			if (selectionMap[i] == 0)
				continue;
			//创建新的未成熟的点
			ImmaturePoint* impt(new ImmaturePoint(x, y, newFrame, selectionMap[i], &Hcalib));

			//插入
			if (!std::isfinite(impt->energyTH))
			{
				delete impt;
			}
			else
			{
				//新建特征点
				Feature* fea(new Feature(Eigen::Vector2d(x, y), 0, 0));
				for (int i = 0; i < MAX_RES_PER_POINT; i++)
					fea->color[i] = impt->color[i];

				fea->mImP = impt;
				impt->mF = fea;
				impt->feaMode = 1;
				fea->_status = Feature::ACTIVE_IMP;

				fea->_frame = newFrame;
				detectorLeft->ComputeDescriptorAndAngle(fea);

				newFrame->immaturePoints.push_back(impt);
				newFrame->_features.push_back(fea);
			}
		}

	// (*mpORBextractorLeft)(newFrame->image, cv::Mat(), newFrame->keypoints, newFrame->descriptors);
	// int numPointsTotal = newFrame->keypoints.size();
	// newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
	// //fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
	// newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
	// newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);

	// for (int i = 0; i < numPointsTotal; i++)
	// {
	// 	ImmaturePoint* impt = new ImmaturePoint(newFrame->keypoints[i].pt.x, newFrame->keypoints[i].pt.y, newFrame, 1, &Hcalib);

	// 	//插入
	// 	if (!std::isfinite(impt->energyTH))
	// 		delete impt;
	// 	else
	// 		newFrame->immaturePoints.push_back(impt);
	// }

	// detectorLeft->Detect(newFrame);
	// //detectorRight->Detect(newFrameRight);
	// //设置新参考帧的点Hessian矩阵

	// int numPointsTotal = newFrame->_features.size();
	// newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
	// //fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
	// newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
	// newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);

	// for (int i = 0; i < numPointsTotal; i++)
	// {
	// 	ImmaturePoint* impt = new ImmaturePoint(newFrame->_features[i]->_pixel[0], newFrame->_features[i]->_pixel[1], newFrame, 1, &Hcalib);

	// 	//插入
	// 	if (!std::isfinite(impt->energyTH))
	// 		delete impt;
	// 	else
	// 		newFrame->immaturePoints.push_back(impt);
	// }

//	std::cout << "t: " << t.elapsed() << std::endl;
	LOG(INFO) << "new features features created: " << (int)newFrame->immaturePoints.size() << " " << (int)newFrame->_features.size() << endl;
}

/**
 * [FullSystem::setPrecalcValues description]
 */
void FullSystem::setPrecalcValues()
{
	for (FrameHessian* fh : frameHessians)
	{
		//每一帧的目标帧大小设为当前帧的大小
		fh->targetPrecalc.resize(frameHessians.size() + 1);
		//设置当前帧的每一个参考帧
		for (unsigned int i = 0; i < frameHessians.size(); i++)
			fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);

		//这里多加了一个目标帧,即当前帧与右帧的目标
		fh->targetPrecalc.back().setStatic(fh, fh->rightFrame, &Hcalib);
	}

	//设置
	ef->setDeltaF(&Hcalib);
}

/**
 * [FullSystem::printLogLine description]
 */
void FullSystem::printLogLine()
{
	if (frameHessians.size() == 0) return;

	if (!setting_debugout_runquiet)
		printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
		       allKeyFramesHistory.back()->id,
		       statistics_lastFineTrackRMSE,
		       ef->resInA,
		       ef->resInL,
		       ef->resInM,
		       (int)statistics_numForceDroppedResFwd,
		       (int)statistics_numForceDroppedResBwd,
		       allKeyFramesHistory.back()->aff_g2l.a,
		       allKeyFramesHistory.back()->aff_g2l.b,
		       frameHessians.back()->shell->id - frameHessians.front()->shell->id,
		       (int)frameHessians.size());


	if (!setting_logStuff) return;

	if (numsLog != 0)
	{
		(*numsLog) << allKeyFramesHistory.back()->id << " "  <<
		           statistics_lastFineTrackRMSE << " "  <<
		           (int)statistics_numCreatedPoints << " "  <<
		           (int)statistics_numActivatedPoints << " "  <<
		           (int)statistics_numDroppedPoints << " "  <<
		           (int)statistics_lastNumOptIts << " "  <<
		           ef->resInA << " "  <<
		           ef->resInL << " "  <<
		           ef->resInM << " "  <<
		           statistics_numMargResFwd << " "  <<
		           statistics_numMargResBwd << " "  <<
		           statistics_numForceDroppedResFwd << " "  <<
		           statistics_numForceDroppedResBwd << " "  <<
		           frameHessians.back()->aff_g2l().a << " "  <<
		           frameHessians.back()->aff_g2l().b << " "  <<
		           frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
		           (int)frameHessians.size() << " "  << "\n";
		numsLog->flush();
	}
}

/**
 * [FullSystem::printEigenValLine description]
 */
void FullSystem::printEigenValLine()
{
	if (!setting_logStuff) return;
	if (ef->lastHS.rows() < 12) return;


	MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
	MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols() - CPARS, ef->lastHS.cols() - CPARS);
	int n = Hp.cols() / 8;
	assert(Hp.cols() % 8 == 0);

	// sub-select
	for (int i = 0; i < n; i++)
	{
		MatXX tmp6 = Hp.block(i * 8, 0, 6, n * 8);
		Hp.block(i * 6, 0, 6, n * 8) = tmp6;

		MatXX tmp2 = Ha.block(i * 8 + 6, 0, 2, n * 8);
		Ha.block(i * 2, 0, 2, n * 8) = tmp2;
	}
	for (int i = 0; i < n; i++)
	{
		MatXX tmp6 = Hp.block(0, i * 8, n * 8, 6);
		Hp.block(0, i * 6, n * 8, 6) = tmp6;

		MatXX tmp2 = Ha.block(0, i * 8 + 6, n * 8, 2);
		Ha.block(0, i * 2, n * 8, 2) = tmp2;
	}

	VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
	VecX eigenP = Hp.topLeftCorner(n * 6, n * 6).eigenvalues().real();
	VecX eigenA = Ha.topLeftCorner(n * 2, n * 2).eigenvalues().real();
	VecX diagonal = ef->lastHS.diagonal();

	std::sort(eigenvaluesAll.data(), eigenvaluesAll.data() + eigenvaluesAll.size());
	std::sort(eigenP.data(), eigenP.data() + eigenP.size());
	std::sort(eigenA.data(), eigenA.data() + eigenA.size());

	int nz = std::max(100, setting_maxFrames * 10);

	if (eigenAllLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
		(*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenAllLog->flush();
	}
	if (eigenALog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
		(*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenALog->flush();
	}
	if (eigenPLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
		(*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		eigenPLog->flush();
	}

	if (DiagonalLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
		(*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		DiagonalLog->flush();
	}

	if (variancesLog != 0)
	{
		VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
		(*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
		variancesLog->flush();
	}

	std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
	(*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
	for (unsigned int i = 0; i < nsp.size(); i++)
		(*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
	(*nullspacesLog) << "\n";
	nullspacesLog->flush();

}

/**
 * [FullSystem::printFrameLifetimes description]
 */
void FullSystem::printFrameLifetimes()
{
	if (!setting_logStuff) return;

	boost::unique_lock<boost::mutex> lock(trackMutex);

	std::ofstream* lg = new std::ofstream();
	lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
	lg->precision(15);

	for (FrameShell* s : allFrameHistory)
	{
		(*lg) << s->id
		      << " " << s->marginalizedAt
		      << " " << s->statistics_goodResOnThis
		      << " " << s->statistics_outlierResOnThis
		      << " " << s->movedByOpt;
		(*lg) << "\n";
	}

	lg->close();
	delete lg;
}

/**
 * [FullSystem::printEvalLine description]
 */
void FullSystem::printEvalLine()
{
	return;
}

}
