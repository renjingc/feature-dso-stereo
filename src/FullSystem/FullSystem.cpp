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
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include "ORB/ORBextractor.h"

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

//void poseEstimationPnP(const Matrix44& T_init,
//	const std::vector<cv::Point3f>& pts_3d_ref_,
//	const std::vector<cv::KeyPoint>& keypoints,
//    const std::vector<cv::DMatch>& feature_matches_,
//	SE3& T_c_r_estimated_)
//{
//    // construct the 3d 2d observations
//    std::vector<cv::Point3f> pts3d;
//    std::vector<cv::Point2f> pts2d;

//    for ( cv::DMatch m:feature_matches_ )
//    {
//        pts3d.push_back( pts_3d_ref_[m.queryIdx] );
//        pts2d.push_back(keypoints[m.trainIdx].pt );
//    }

//    cv::Mat K;
//    cv::eigen2cv(_K,K);

//    Eigen::Matrix<float, 3, 3> r;
//    Eigen::Vector3f t;
//    r=T_init.block<3,3>(0,0);
//    t<<T_init(0,3),T_init(1,3),T_init(2,3);
//    cv::Mat rvec, tvec, inliers;

//    cv::eigen2cv(r,rvec);
//    cv::eigen2cv(t,tvec);
//    cv::solvePnPRansac( pts3d, pts2d, K, Mat(), rvec, tvec, true, 100, 4.0, 0.99, inliers,EPNP);
//    int num_inliers_ = inliers.rows;

////    int inlierCount=0;
////    ransac_cc(pts2d,pts3d,K,Mat(),rvec,tvec,inlierCount);
////    num_inliers_=inlierCount;

//    cout<<"pnp inliers: "<<num_inliers_<<endl;
//    T_c_r_estimated_ = SE3(
//        SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)),
//        Vector3d( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0))
//    );

//    // using bundle adjustment to optimize the pose
//    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6,2>> Block;
//    Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
//    Block* solver_ptr = new Block( linearSolver );
//    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( solver_ptr );
//    g2o::SparseOptimizer optimizer;
//    optimizer.setAlgorithm ( solver );

//    g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
//    pose->setId ( 0 );
//    pose->setEstimate ( g2o::SE3Quat (
//                            T_c_r_estimated_.rotation_matrix(), T_c_r_estimated_.translation()
//                        ) );
//    optimizer.addVertex ( pose );

//    // edges
//    for ( int i=0; i<inliers.rows; i++ )
//    {
//        int index = inliers.at<int>(i,0);
//        // 3D -> 2D projection
//        EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
//        edge->setId(i);
//        edge->setVertex(0, pose);
//        edge->camera_ = camera_.get();
//        edge->point_ = Vector3d( pts3d[index].x, pts3d[index].y, pts3d[index].z );
//        edge->setMeasurement( Vector2d(pts2d[index].x, pts2d[index].y) );
//        edge->setInformation( Eigen::Matrix2d::Identity() );
//        optimizer.addEdge( edge );
//    }

//    optimizer.initializeOptimization();
//    optimizer.optimize(10);

//    T_c_r_estimated_ = SE3 (
//        pose->estimate().rotation(),
//        pose->estimate().translation()
//    );
//}
/**
 *
 */
FullSystem::FullSystem(): matcher_flann_(new cv::flann::LshIndexParams(5, 10, 2))
{
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

	mpORBextractorLeft = new ORBextractor(300, 1.2, 3, 20, 8);
	mpORBextractorRight = new ORBextractor(300, 1.2, 3, 20, 8);
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
	for (FrameHessian* fh : unmappedTrackedFrames)
		delete fh;

	delete coarseDistanceMap;
	delete coarseTracker;
	delete coarseTracker_forNewKF;
	delete coarseInitializer;
	delete pixelSelector;
	delete ef;
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

	Eigen::Matrix<double, 3, 3> last_R = (*(allFrameHistory.begin()))->camToWorld.so3().matrix();
	Eigen::Matrix<double, 3, 1> last_T = (*(allFrameHistory.begin()))->camToWorld.translation().transpose();

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
 * @param  fh_right [description]
 * @return          [description]
 */
Vec4 FullSystem::trackNewCoarse(FrameHessian* fh, FrameHessian* fh_right, Eigen::Matrix3d R)
{

	assert(allFrameHistory.size() > 0);
	// set pose initialization.

//    printf("the size of allFrameHistory is %d \n", (int)allFrameHistory.size());

	// show original images
	for (IOWrap::Output3DWrapper* ow : outputWrapper)
	{
		ow->pushStereoLiveFrame(fh, fh_right);
	}

	Sophus::SO3 init_R(R);
	Eigen::Vector3d t(0, 0, 0); //
	Sophus::SE3 init_Rt(R, t); //

	//参考帧
	FrameHessian* lastF = coarseTracker->lastRef;

	//a和b
	AffLight aff_last_2_l = AffLight(0, 0);

	//
	std::vector<SE3, Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;

	// for first two frames process differently
	if (allFrameHistory.size() == 2)
	{
		initializeFromInitializer(fh);

		lastF_2_fh_tries.push_back(init_Rt);
		lastF_2_fh_tries.push_back(SE3(R, Eigen::Matrix<double, 3, 1>::Zero() ));

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

		boost::timer timer;
		//是否跟踪成功
		bool trackingIsGood = coarseTracker->trackNewestCoarse(
		                        fh, lastF_2_fh_this, aff_g2l_this,
		                        pyrLevelsUsed - 1,
		                        achievedRes);	// in each level has to be at least as good as the last try.
		std::cout << "trackNewestCoarse: " << timer.elapsed() << std::endl;

		//尝试次数++
		tryIterations++;

		if (i != 0)
		{
			printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
			       i,
			       i, pyrLevelsUsed - 1,
			       aff_g2l_this.a, aff_g2l_this.b,
			       achievedRes[0],
			       achievedRes[1],
			       achievedRes[2],
			       achievedRes[3],
			       achievedRes[4],
			       coarseTracker->lastResiduals[0],
			       coarseTracker->lastResiduals[1],
			       coarseTracker->lastResiduals[2],
			       coarseTracker->lastResiduals[3],
			       coarseTracker->lastResiduals[4]);
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
		printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
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
	std::cout << "x:" << last_T(0, 0) << "y:" << last_T(1, 0) << "z:" << last_T(2, 0) << std::endl;

	if (coarseTracker->firstCoarseRMSE < 0)
		coarseTracker->firstCoarseRMSE = achievedRes[0];

	if (!setting_debugout_runquiet)
		printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);

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

/**
 * [FullSystem::stereoMatch description]
 * @param image       [description]
 * @param image_right [description]
 * @param id          [description]
 * @param idepthMap   [description]
 */
void FullSystem::stereoMatch( ImageAndExposure* image, ImageAndExposure* image_right, int id, cv::Mat &idepthMap)
{
	// =========================== add into allFrameHistory =========================
	FrameHessian* fh = new FrameHessian();
	FrameHessian* fh_right = new FrameHessian();
	FrameShell* shell = new FrameShell();
	shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
	shell->aff_g2l = AffLight(0, 0);
	shell->marginalizedAt = shell->id = allFrameHistory.size();
	shell->timestamp = image->timestamp;

	//一直递增的id
	shell->incoming_id = id; // id passed into DSO

	//当前帧的信息
	fh->shell = shell;
	fh_right->shell = shell;

	// =========================== make Images / derivatives etc. =========================
	fh->ab_exposure = image->exposure_time;

	//左图的梯度点
	fh->makeImages(image->image, &Hcalib);
	fh_right->ab_exposure = image_right->exposure_time;

	//右图的梯度点
	fh_right->makeImages(image_right->image, &Hcalib);

	//内参
	Mat33f K = Mat33f::Identity();
	K(0, 0) = Hcalib.fxl();
	K(1, 1) = Hcalib.fyl();
	K(0, 2) = Hcalib.cxl();
	K(1, 2) = Hcalib.cyl();

	//计数
	int counter = 0;

	//创建新的一帧中的点
	makeNewTraces(fh, fh_right, 0);

	//逆深度图
	unsigned  char * idepthMapPtr = idepthMap.data;

	//遍历每一个点
	for (ImmaturePoint* ph : fh->immaturePoints)
	{
		//坐标
		ph->u_stereo = ph->u;
		ph->v_stereo = ph->v;
		ph->idepth_min_stereo = ph->idepth_min = 0;
		ph->idepth_max_stereo = ph->idepth_max = NAN;

		//左图与右图进行双目匹配，获取静态深度
		ImmaturePointStatus phTraceRightStatus = ph->traceStereo(fh_right, K, 1);

		//判断当前点的深度值是否好
		if (phTraceRightStatus == ImmaturePointStatus::IPS_GOOD)
		{
			//获取右图中的这个点
			ImmaturePoint* phRight = new ImmaturePoint(ph->lastTraceUV(0), ph->lastTraceUV(1), fh_right, &Hcalib );

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

				//更新逆深度图
				*((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u * 3) = ph->idepth_stereo;
				*((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u * 3 + 1) = ph->idepth_min;
				*((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u * 3 + 2) = ph->idepth_max;

				counter++;
			}
		}
	}

//    std::sort(error.begin(), error.end());
//    std::cout << 0.25 <<" "<<error[error.size()*0.25].first<<" "<<
//              0.5 <<" "<<error[error.size()*0.5].first<<" "<<
//              0.75 <<" "<<error[error.size()*0.75].first<<" "<<
//              0.1 <<" "<<error.back().first << std::endl;

//    for(int i = 0; i < error.size(); i++)
//        std::cout << error[i].first << " " << error[i].second.first << " " << error[i].second.second << std::endl;

	std::cout << " frameID " << id << " got good matches " << counter << std::endl;

	delete fh;
	delete fh_right;

	return;
}

// process nonkey frame to refine key frame idepth
/**
 * [FullSystem::traceNewCoarseNonKey description]
 * @param fh       [description]
 * @param fh_right [description]
 */
void FullSystem::traceNewCoarseNonKey(FrameHessian *fh, FrameHessian *fh_right)
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

	//遍历每一帧
	for (FrameHessian *host : frameHessians)        // go through all active frames
	{
		//个数
		// number++;
		int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

		//参考帧到当前帧位姿
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
		for (ImmaturePoint *ph : host->immaturePoints)
		{
			//进行点的跟踪
			// do temperol stereo match
			ImmaturePointStatus phTrackStatus = ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

			//如果是好的点
			if (phTrackStatus == ImmaturePointStatus::IPS_GOOD)
			{
				//新建一个点
				ImmaturePoint *phNonKey = new ImmaturePoint(ph->lastTraceUV(0), ph->lastTraceUV(1), fh, &Hcalib);

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
				ImmaturePointStatus phNonKeyStereoStatus = phNonKey->traceStereo(fh_right, K, 1);

				//静态状态是好的
				if (phNonKeyStereoStatus == ImmaturePointStatus::IPS_GOOD)
				{
					//右边点
					ImmaturePoint* phNonKeyRight = new ImmaturePoint(phNonKey->lastTraceUV(0), phNonKey->lastTraceUV(1), fh_right, &Hcalib );

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
						ph->lastTraceStatus = ImmaturePointStatus :: IPS_OUTLIER;
						continue;
					}
					else
					{
						//重投影该点，更新最小和最大逆深度
						// project back
						Vec3f pinverse_min = KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) / phNonKey->idepth_min_stereo - t);
						idepth_min_update = 1.0f / pinverse_min(2);

						Vec3f pinverse_max = KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) / phNonKey->idepth_max_stereo - t);
						idepth_max_update = 1.0f / pinverse_max(2);

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
 * @param fh_right [description]
 */
void FullSystem::traceNewCoarseKey(FrameHessian* fh, FrameHessian* fh_right)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

	//内参
	Mat33f K = Mat33f::Identity();
	K(0, 0) = Hcalib.fxl();
	K(1, 1) = Hcalib.fyl();
	K(0, 2) = Hcalib.cxl();
	K(1, 2) = Hcalib.cyl();

	//遍历每一个关键帧的中的点
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
 */
void FullSystem::activatePointsMT_Reductor(
  std::vector<PointHessian*>* optimized,
  std::vector<ImmaturePoint*>* toOptimize,
  int min, int max, Vec10* stats, int tid)
{
	ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];

	//优化每一个点
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
			//判断点的状态，删除未成功跟踪的点，或删除最后一个跟踪点上的离群点。
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
			// 这个点是否能激活
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

	//多线程优化每一个点的逆深度
	if (multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);
	else
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);

	//遍历每一个优化前的点
	for (unsigned k = 0; k < toOptimize.size(); k++)
	{
		//该点优化后的
		PointHessian* newpoint = optimized[k];
		ImmaturePoint* ph = toOptimize[k];

		//新的点好的
		if (newpoint != 0 && newpoint != (PointHessian*)((long)(-1)))
		{
			//新的点
			newpoint->host->immaturePoints[ph->idxInImmaturePoints] = 0;

			//该点的主导帧的点插入该点
			newpoint->host->pointHessians.push_back(newpoint);

			//误差函数中加入该点
			ef->insertPoint(newpoint);

			//遍历每一个点与帧的残差，ef中插入该残差
			for (PointFrameResidual* r : newpoint->residuals)
				ef->insertResidual(r);
			assert(newpoint->efPoint != 0);
			delete ph;
		}
		else if (newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus == IPS_OOB)
		{
			//删除该点
			delete ph;
			ph->host->immaturePoints[ph->idxInImmaturePoints] = 0;
		}
		else
		{
			assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
		}
	}

	//遍历每一个主导帧
	for (FrameHessian* host : frameHessians)
	{
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
 * efPoint->stateFlag = EFPointStatus::PS_DROP
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
						r->linearize(&Hcalib);
						r->efResidual->isLinearized = false;
						r->applyRes(true);
						if (r->efResidual->isActive())
						{
							r->efResidual->fixLinearizationF(ef);
							ngoodRes++;
						}
					}
					if (ph->idepth_hessian > setting_minIdepthH_marg)
					{
						flag_inin++;
						ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
						host->pointHessiansMarginalized.push_back(ph);
					}
					else
					{
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
						host->pointHessiansOut.push_back(ph);
					}

				}
				else
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
	}
}


void FullSystem::ExtractORB(int flag, const cv::Mat &im)
{
	if (flag == 0)
		(*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);
	else
		(*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight);
}
void FullSystem::find_feature_matches (const cv::Mat& descriptorsLast, const cv::Mat& descriptorsCur, std::vector<cv::DMatch>& feature_matches_)
{
	if (descriptorsLast.empty() || descriptorsCur.empty())
	{
		std::cout << "error" << std::endl;
		return;
	}
	std::vector<cv::DMatch> matches;
	matcher_flann_.match( descriptorsLast, descriptorsCur, matches );
	// select the best matches
	float min_dis = std::min_element (
	                  matches.begin(), matches.end(),
	                  [] ( const cv::DMatch & m1, const cv::DMatch & m2 )
	{
		return m1.distance < m2.distance;
	} )->distance;

	feature_matches_.clear();
	for ( cv::DMatch& m : matches )
	{
		if ( m.distance < std::max<float> ( min_dis * 2.0, 30.0 ) )
		{
			feature_matches_.push_back(m);
		}
	}
}

// bool checkEstimatedPose(const cv::Mat& R,const cv::Mat& t,)
// {
//           // check if the estimated pose is good
//           if ( num_inliers_ < min_inliers_ )
//           {
//               std::cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
//               return false;
//           }
//           // if the motion is too large, it is probably wrong
//           Sophus::Vector6d d = T_c_r_estimated_.log();
//           if ( d.norm() > 3.0 )
//           {
//               std::cout<<"reject because motion is too large: "<<d.norm()<<endl;
//               return false;
//           }
//           return true;
// }
//
//
int FullSystem::CheckFrameDescriptors (
  FrameShell* frame1,
  FrameShell* frame2,
  std::list<std::pair<int, int>>& matches
)
{
	std::vector<int> distance;
	for ( auto& m : matches )
	{
		distance.push_back( DescriptorDistance(
		                      frame1->descriptorsLeft,
		                      frame2->_descriptorsLeft
		                    ));
	}

	int cnt_good = 0;
	int best_dist = *std::min_element( distance.begin(), distance.end() );
	//LOG(INFO) << "best dist = " << best_dist << endl;

	// 取个上下限
	best_dist = best_dist > _options.init_low ? best_dist : _options.init_low;
	best_dist = best_dist < _options.init_high ? best_dist : _options.init_high;

	int i = 0;
	//LOG(INFO) << "original matches: " << matches.size() << endl;
	for ( auto iter = matches.begin(); iter != matches.end(); i++ )
	{
		if ( distance[i] < _options.initMatchRatio * best_dist )
		{
			cnt_good++;
			iter++;
		}
		else
		{
			iter = matches.erase( iter );
		}
	}
	//LOG(INFO) << "correct matches: " << matches.size() << endl;
	return cnt_good;
}

int FullSystem::DescriptorDistance ( const cv::Mat& a, const cv::Mat& b )
{
	const int *pa = a.ptr<int32_t>();
	const int *pb = b.ptr<int32_t>();
	int dist = 0;
	for (int i = 0; i < 8; i++, pa++, pb++)
	{
		unsigned  int v = *pa ^ *pb;
		v = v - ((v >> 1) & 0x55555555);
		v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
		dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
	}
	return dist;
}
void FullSystem::ComputeBoW(FrameShell* f)
{
	if ( _vocab != nullptr && f->_bow_vec.empty() )
	{
		_vocab->transform( f->descriptorsLeft, f->_bow_vec, f->_feature_vec, 4);
	}
}
/**
 * [FullSystem::addActiveFrame description]
 * @param image       [description]
 * @param image_right [description]
 * @param id          [description]
 */
void FullSystem::addActiveFrame( ImageAndExposure* image, ImageAndExposure* image_right, int id )
{
	if (isLost) return;
	boost::unique_lock<boost::mutex> lock(trackMutex);

	// =========================== add into allFrameHistory =========================
	//新建一个帧Hessian类
	FrameHessian* fh = new FrameHessian();
	//新建一个帧的位姿信息
	FrameHessian* fh_right = new FrameHessian();
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
	fh_right->shell = shell;

	// =========================== make Images / derivatives etc. =========================
	//曝光时间
	fh->ab_exposure = image->exposure_time;
	//得到当前帧的每一层的灰度图像和xy方向梯度值和xy梯度平方和，用于跟踪和初始化
	fh->makeImages(image->image, &Hcalib);
	//曝光时间
	fh_right->ab_exposure = image_right->exposure_time;
	//得到当前帧的每一层的灰度图像和xy方向梯度值和xy梯度平方和，用于跟踪和初始化

	fh_right->makeImages(image_right->image, &Hcalib);


	cv::Mat imageLeft = cv::Mat(image->h, image->w, CV_32FC1, image->image);
	cv::Mat imageRight = cv::Mat(image->h, image->w, CV_32FC1, image_right->image);
	cv::Mat imageLeft8, imageRight8;
	imageLeft.convertTo(imageLeft8, CV_8U, 1, 0);
	imageRight.convertTo(imageRight8, CV_8U, 1, 0);
	//提取特征点
	// ORB extraction
	// 同时对左右目提特征

	(*mpORBextractorLeft)(imageLeft8, cv::Mat(), mvKeys, mDescriptors);
	(*mpORBextractorRight)(imageRight8, cv::Mat(), mvKeysRight, mDescriptorsRight);

	shell->keypointsLeft = mvKeys;
	shell->keypointsRight = mvKeysRight;
	mDescriptors.copyTo(shell->descriptorsLeft);
	mDescriptorsRight.copyTo(shell->descriptorsRight);
	imageLeft8.copyTo(shell->imageLeft);
	imageRight8.copyTo(shell->imageRight);

	// for(int i=0;i<mvKeys.size();i++)
	// 	cv::circle(imageLeft8, mvKeys[i].pt, 2, cv::Scalar(255,0,0),2);
	// for(int i=0;i<mvKeysRight.size();i++)
	// 	cv::circle(imageRight8, mvKeysRight[i].pt, 2, cv::Scalar(255,0,0),2);

	// imshow("imageLeft",imageLeft8);
	// imshow("imageRight",imageRight8);

	//将当前帧增加到历史记录中
	allFrameHistory.push_back(shell);

	if (!initialized)
	{
		// use initializer!
		if (coarseInitializer->frameID < 0)	// first frame set. fh is kept by coarseInitializer.
		{
			//设置初始的双目
			coarseInitializer->setFirstStereo(&Hcalib, fh, fh_right);
			//初始化成功
			initialized = true;
		}
		return;
	}
	else	// do front-end operation.
	{
		Eigen::Matrix3d initR;
		std::vector<cv::DMatch> matches;
		cv::Mat image;

		ComputeBoW(allFrameHistory[allFrameHistory.size() - 2]);
		ComputeBoW(allFrameHistory[allFrameHistory.size() - 1]);

		//find_feature_matches(allFrameHistory[allFrameHistory.size() - 2]->descriptorsLeft, mDescriptors, matches);

		if (matches.size() > 5)
		{
			cv::Mat K = (cv::Mat_<float>(3, 3) << Hcalib.fxl(), 0, Hcalib.cxl(), 0, Hcalib.fyl(), Hcalib.cyl(), 0, 0, 1);
			std::vector<cv::Point2f> points1;
			std::vector<cv::Point2f> points2;
			for (int i = 0; i < (int)matches.size(); i++)
			{
				points1.push_back(allFrameHistory[allFrameHistory.size() - 2]->keypointsLeft[matches[i].queryIdx].pt);
				points2.push_back(mvKeys[matches[i].trainIdx].pt);
			}
			// std::cout<<Hcalib.fxl()<<" "<<Hcalib.fyl()<<" "<<Hcalib.cxl()<<" "<<Hcalib.cyl()<<std::endl;
			cv::Point2d principal_point(Hcalib.cxl(), Hcalib.cyl());
			float focal_length = Hcalib.fxl();
			cv::Mat R, t;
			cv::Mat essential_matrix = cv::findEssentialMat(points1, points2, K, CV_RANSAC);
			cv::recoverPose(essential_matrix, points1, points2, K, R, t);
			cv::cv2eigen(R, initR);
			// std::cout << initR << std::endl;
		}
		// cv::drawMatches(allFrameHistory[allFrameHistory.size()-2]->imageLeft,allFrameHistory[allFrameHistory.size()-2]->keypointsLeft,
		//              	imageLeft8,mvKeys,
		//           	matches,
		//           	image,
		//           	cv::Scalar(255,0,0)
		//           );
		// cv::imshow("match",image);

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

		Vec4 tres = trackNewCoarse(fh, fh_right, initR);
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
			printf(" delta is %f \n", delta);
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
		deliverTrackedFrame(fh, fh_right, needToMakeKF);
		return;
	}
}

/**
 * [FullSystem::deliverTrackedFrame description]
 * @param fh       [description]
 * @param fh_right [description]
 * @param needKF   [description]
 */
void FullSystem::deliverTrackedFrame(FrameHessian* fh, FrameHessian* fh_right, bool needKF)
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
		if (needKF) makeKeyFrame(fh, fh_right);
		//不加入关键帧
		else makeNonKeyFrame(fh, fh_right);
	}
	//如果不使用linearizeOperation
	else
	{
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
		//不建图序列插入
		unmappedTrackedFrames.push_back(fh);
		unmappedTrackedFrames_right.push_back(fh_right);
		//是否是关键帧，设置最新的关键帧id
		if (needKF)
			needNewKFAfter = fh->shell->trackingRef->id;

		//通知处在等待该对象的线程的方法
		//唤醒所有正在等待该对象的线程
		trackedFrameSignal.notify_all();

		//当跟踪的参考帧为-1时，则建图信号上锁
		while (coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 )
		{
			mappedFrameSignal.wait(lock);
		}

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
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

	while (runMapping)
	{
		//若unmappedTrackedFrames为０
		while (unmappedTrackedFrames.size() == 0)
		{
			//跟踪线程进行等待
			trackedFrameSignal.wait(lock);
			if (!runMapping) return;
		}

		//最前面的一帧
		FrameHessian* fh = unmappedTrackedFrames.front();
		//弹出
		unmappedTrackedFrames.pop_front();
		FrameHessian* fh_right = unmappedTrackedFrames_right.front();
		unmappedTrackedFrames_right.pop_front();

		// guaranteed to make a KF for the very first two tracked frames.
		// 保证为前两个跟踪帧制作一个KF。小于一帧的时候
		if (allKeyFramesHistory.size() <= 2)
		{
			lock.unlock();
			makeKeyFrame(fh, fh_right);
			lock.lock();
			mappedFrameSignal.notify_all();
			continue;
		}

		//unmappedTrackedFrames>3的时，需要进行needToKetchupMapping
		if (unmappedTrackedFrames.size() > 3)
			needToKetchupMapping = true;

		//needToKetchupMapping大于0
		if (unmappedTrackedFrames.size() > 0) // if there are other frames to track, do that first.
		{
			//插入非关键帧
			lock.unlock();
			makeNonKeyFrame(fh, fh_right);
			lock.lock();

			if (needToKetchupMapping && unmappedTrackedFrames.size() > 0)
			{
				//弹出最前面的一帧
				FrameHessian* fh = unmappedTrackedFrames.front();
				unmappedTrackedFrames.pop_front();
				{
					boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
					assert(fh->shell->trackingRef != 0);

					//当前帧位姿
					fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
					fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
				}
				delete fh;
				delete fh_right;
			}
		}
		else
		{
			//setting_realTimeMaxKF=false实时最大帧数
			//若当前最新的关键帧id大于关键帧序列中最后的一帧
			if (setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id)
			{
				//插入关键帧
				lock.unlock();
				makeKeyFrame(fh, fh_right);
				//不进行KetchupMapping
				needToKetchupMapping = false;
				lock.lock();
			}
			else
			{
				//插入非关键帧
				lock.unlock();
				makeNonKeyFrame(fh, fh_right);
				lock.lock();
			}
		}
		mappedFrameSignal.notify_all();
	}
	printf("MAPPING FINISHED!\n");
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
	//解锁
	lock.unlock();
	//mapping线程阻塞
	mappingThread.join();
}

/**
 * [FullSystem::makeNonKeyFrame description]
 * @param fh       [description]
 * @param fh_right [description]
 * 不将当前帧作为关键帧
 */
void FullSystem::makeNonKeyFrame( FrameHessian* fh, FrameHessian* fh_right)
{
	// needs to be set by mapping thread. no lock required since we are in mapping thread.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		//根据参考帧到世界坐标系的变换和当前帧和参考帧之间的变换，计算当前帧到世界坐标系的变换
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		//设置当前帧的位姿和a和b
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
	}

	//跟踪这一帧中的每个点，对这个点的像素坐标和状态进行更新
	traceNewCoarseNonKey(fh, fh_right);

	delete fh;
	delete fh_right;
}

/**
 * [FullSystem::makeKeyFrame description]
 * @param fh       [description]
 * @param fh_right [description]
 *
 * 1.先对每个点进行更新
 * 2.判断窗口中的关键帧，是否边缘化关键帧
 * 3.设置每一个关键帧之间为主导帧
 * 4.加入每一个关键帧中的点与其它关键帧的残差
 * 5.遍历窗口中的每一个关键帧的每一个点，判断这个点的状态并且将这个点与每一个关键帧进行逆深度残差更新，更新该点的逆深度
 * 	并在ef中插入该点，加入该点与每一个关键帧的残差
 * 6.优化，最大优化次数6次
 * 7.移除外点removeOutliers
 * 8.设置新的跟踪器coarseTracker_forNewKF
 * 9.删除点，并在ef中删除点和并跟新ef中的H和b
 */
void FullSystem::makeKeyFrame( FrameHessian* fh, FrameHessian* fh_right)
{
	// needs to be set by mapping thread
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
		//根据参考帧到世界坐标系的变换和当前帧和参考帧之间的变换，计算当前帧到世界坐标系的变换
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		//设置当前帧的位姿和a和b
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
	}

	//跟踪这一帧中的每个点，对这个点的像素坐标和状态进行更新
	traceNewCoarseKey(fh, fh_right);

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

	//关键帧id
	fh->frameID = allKeyFramesHistory.size();

	//插入关键帧
	allKeyFramesHistory.push_back(fh->shell);

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

	// =========================== Activate Points (& flag for marginalization). =========================
	// 遍历窗口中的每一个关键帧的每一个点，判断这个点的状态并且将这个点与每一个关键帧进行逆深度残差更新，更新该点的逆深度
	// 并在ef中插入该点，加入该点与每一个关键帧的残差
	activatePointsMT();
	ef->makeIDX();

	// =========================== OPTIMIZE ALL =========================
	//每一帧的误差阈值
	fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;

	//优化
	float rmse = optimize(setting_maxOptIterations);

	//printf("allKeyFramesHistory size is %d \n", (int)allKeyFramesHistory.size());
	printf("rmse is %f \n", rmse);

	// =========================== Figure Out if INITIALIZATION FAILED =========================
	//判断初始化是否成功
	if (allKeyFramesHistory.size() <= 4)
	{
		if (allKeyFramesHistory.size() == 2 && rmse > 20 * benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed = true;
		}
		if (allKeyFramesHistory.size() == 3 && rmse > 13 * benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed = true;
		}
		if (allKeyFramesHistory.size() == 4 && rmse > 9 * benchmark_initializerSlackFactor)
		{
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed = true;
		}
	}

	if (isLost)
		return;

	// =========================== REMOVE OUTLIER =========================
	//移除外点
	removeOutliers();
	{
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
		//设置新的跟踪器的内参
		coarseTracker_forNewKF->makeK(&Hcalib);
		//设置新的跟踪器的参考帧，并且使用双目静态匹配获取参考帧的点的逆深度
		coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians, fh_right, Hcalib);

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
	//获取当前新的关键帧的点
	makeNewTraces(fh, fh_right, 0);

	//发布关键帧
	for (IOWrap::Output3DWrapper* ow : outputWrapper)
	{
		ow->publishGraph(ef->connectivityMap);
		ow->publishKeyframes(frameHessians, false, &Hcalib);
	}

	// =========================== Marginalize Frames =========================
	//边缘化帧
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

	delete fh_right;

//	printLogLine();
//    printEigenValLine();

}

// insert the first Frame into FrameHessians
/**
 * [FullSystem::initializeFromInitializer description]
 * @param newFrame [description]
 */
void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
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
	//能量函数插入当前帧
	ef->insertFrame(firstFrame, &Hcalib);
	//设置每一帧的目标帧，这时候只有第一帧
	setPrecalcValues();

	//第一帧的右帧
	FrameHessian* firstFrameRight = coarseInitializer->firstRightFrame;
	//
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

	if (!setting_debugout_runquiet)
		printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100 * keepPercentage,
		       (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0] );

	// initialize first frame by idepth computed by static stereo matching
	// 遍历每一个点
	for (int i = 0; i < coarseInitializer->numPoints[0]; i++)
	{
		if (rand() / (float)RAND_MAX > keepPercentage) continue;

		Pnt* point = coarseInitializer->points[0] + i;

		//初始化一个点
		ImmaturePoint* pt = new ImmaturePoint(point->u + 0.5f, point->v + 0.5f, firstFrame, point->my_type, &Hcalib);

		//设置该点的坐标和最小和最大的逆深度
		pt->u_stereo = pt->u;
		pt->v_stereo = pt->v;
		pt->idepth_min_stereo = 0;
		pt->idepth_max_stereo = NAN;

		//静态双目跟踪，左图与右图进行匹配
		pt->traceStereo(firstFrameRight, K, 1);

		//设置点的最小和最大逆深度
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
		PointHessian* ph = new PointHessian(pt, &Hcalib);
		delete pt;
		if (!std::isfinite(ph->energyTH)) {delete ph; continue;}

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
	printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

/**
 * [FullSystem::makeNewTraces description]
 * @param newFrame      [description]
 * @param newFrameRight [description]
 * @param gtDepth       [description]
 * 选取新关键帧的点
 */
void FullSystem::makeNewTraces(FrameHessian* newFrame, FrameHessian* newFrameRight, float* gtDepth)
{
	pixelSelector->allowFast = true;
	//int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	//筛选新的点，点的总数
	int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap, setting_desiredImmatureDensity);

	//设置新参考帧的点Hessian矩阵
	newFrame->pointHessians.reserve(numPointsTotal * 1.2f);
	//fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansMarginalized.reserve(numPointsTotal * 1.2f);
	newFrame->pointHessiansOut.reserve(numPointsTotal * 1.2f);

	//遍历每一个点
	for (int y = patternPadding + 1; y < hG[0] - patternPadding - 2; y++)
		for (int x = patternPadding + 1; x < wG[0] - patternPadding - 2; x++)
		{
			int i = x + y * wG[0];
			if (selectionMap[i] == 0)
				continue;
			//创建新的未成熟的点
			ImmaturePoint* impt = new ImmaturePoint(x, y, newFrame, selectionMap[i], &Hcalib);

			//插入
			if (!std::isfinite(impt->energyTH))
				delete impt;
			else
				newFrame->immaturePoints.push_back(impt);
		}
	printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());

}

/**
 * [FullSystem::setPrecalcValues description]
 */
void FullSystem::setPrecalcValues()
{
	for (FrameHessian* fh : frameHessians)
	{
		//每一帧的目标帧大小设为当前帧的大小
		fh->targetPrecalc.resize(frameHessians.size());
		//设置当前帧的每一个参考帧
		for (unsigned int i = 0; i < frameHessians.size(); i++)
			fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
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
