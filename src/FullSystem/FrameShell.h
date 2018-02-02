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
#include "algorithm"

#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/eigen.hpp>



namespace fdso
{
class FrameHessian;
class PointHessian;

/**
 * @brief      Class for frame shell.
 * 每一帧的信息
 */
class FrameShell
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

	//一直递增的id
	int id; 			// INTERNAL ID, starting at zero.

	//插入dso的id
	int incoming_id;	// ID passed into DSO
	//时间戳
	double timestamp;		// timestamp passed into DSO.

	// set once after tracking
	//相对与参考帧的变换
	SE3 camToTrackingRef;
	//参考帧的信息
	FrameShell* trackingRef;

	// constantly adapted.
	//相对于世界坐标系的变换
	SE3 camToWorld;
	SE3 camToWorldOpti;
	// Write: TRACKING, while frame is still fresh; MAPPING: only when locked [shellPoseMutex].
	//a和ｂ
	AffLight aff_g2l;
	//位姿的有效性
	bool poseValid;

	// statisitcs
	//out的
	int statistics_outlierResOnThis;
	//good
	int statistics_goodResOnThis;

	//边缘
	int marginalizedAt;

	//
	double movedByOpt;

	inline FrameShell()
	{
		id=0;
		poseValid=true;
		camToWorld = SE3();
		camToWorldOpti = SE3();
		timestamp=0;
		marginalizedAt=-1;
		movedByOpt=0;
		statistics_outlierResOnThis=statistics_goodResOnThis=0;
		trackingRef=0;
		camToTrackingRef = SE3();
	}

};


}

