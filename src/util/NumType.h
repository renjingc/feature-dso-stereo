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

#include "sophus/sim3.hpp"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"

#include <glog/logging.h>

#include <Eigen/StdVector>

// for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
using Eigen::Vector2d;
using Eigen::Vector3d;

using Sophus::SO3;
using Sophus::SE3;

// for cv
#include <opencv2/core/core.hpp>
using cv::Mat;

// std
#include <vector>
#include <list>
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <unordered_map>
#include <map>

#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"

using namespace std;


namespace fdso
{

// CAMERA MODEL TO USE


#define SSEE(val,idx) (*(((float*)&val)+idx))

#define MAX_RES_PER_POINT 8
#define NUM_THREADS 6

#define todouble(x) (x).cast<double>()


typedef Sophus::SE3d SE3;
typedef Sophus::Sim3d Sim3;
typedef Sophus::SO3d SO3;


#define CPARS 4 //-- Calibration Parameters


typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;
typedef Eigen::Matrix<double, CPARS, CPARS> MatCC;
#define MatToDynamic(x) MatXX(x)


typedef Eigen::Matrix<double, CPARS, 10> MatC10;
typedef Eigen::Matrix<double, 10, 10> Mat1010;
typedef Eigen::Matrix<double, 13, 13> Mat1313;

typedef Eigen::Matrix<double, 8, 10> Mat810;
typedef Eigen::Matrix<double, 8, 3> Mat83;
typedef Eigen::Matrix<double, 6, 6> Mat66;
typedef Eigen::Matrix<double, 5, 3> Mat53;
typedef Eigen::Matrix<double, 4, 3> Mat43;
typedef Eigen::Matrix<double, 4, 2> Mat42;
typedef Eigen::Matrix<double, 3, 3> Mat33;
typedef Eigen::Matrix<double, 2, 2> Mat22;
typedef Eigen::Matrix<double, 8, CPARS> Mat8C;
typedef Eigen::Matrix<double, CPARS, 8> MatC8;
typedef Eigen::Matrix<float, 8, CPARS> Mat8Cf;
typedef Eigen::Matrix<float, CPARS, 8> MatC8f;

typedef Eigen::Matrix<double, 10, CPARS> Mat10C;
typedef Eigen::Matrix<double, CPARS, 10> MatC10;
typedef Eigen::Matrix<float, 10, CPARS> Mat10Cf;
typedef Eigen::Matrix<float, CPARS, 10> MatC10f;

typedef Eigen::Matrix<double, 8, 8> Mat88;
typedef Eigen::Matrix<double, 7, 7> Mat77;

typedef Eigen::Matrix<double, CPARS, 1> VecC;
typedef Eigen::Matrix<float, CPARS, 1> VecCf;
typedef Eigen::Matrix<double, 13, 1> Vec13;
typedef Eigen::Matrix<double, 10, 1> Vec10;
typedef Eigen::Matrix<double, 9, 1> Vec9;
typedef Eigen::Matrix<double, 8, 1> Vec8;
typedef Eigen::Matrix<double, 7, 1> Vec7;
typedef Eigen::Matrix<double, 6, 1> Vec6;
typedef Eigen::Matrix<double, 5, 1> Vec5;
typedef Eigen::Matrix<double, 4, 1> Vec4;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;

typedef Eigen::Matrix<float, 3, 3> Mat33f;
typedef Eigen::Matrix<float, 10, 3> Mat103f;
typedef Eigen::Matrix<float, 2, 2> Mat22f;
typedef Eigen::Matrix<float, 15, 1> Vec15f;
typedef Eigen::Matrix<float, 3, 1> Vec3f;
typedef Eigen::Matrix<float, 2, 1> Vec2f;
typedef Eigen::Matrix<float, 6, 1> Vec6f;

typedef Eigen::AngleAxisd AngleAxis;
typedef Eigen::Matrix<double, 9, 1> SpeedAndBias;
typedef Eigen::Matrix<double, 15, 15> covariance_t;
typedef Eigen::Matrix<double, 15, 15> jacobian_t;

typedef Eigen::Matrix<double, 4, 9> Mat49;
typedef Eigen::Matrix<double, 8, 9> Mat89;

typedef Eigen::Matrix<double, 9, 4> Mat94;
typedef Eigen::Matrix<double, 9, 8> Mat98;

typedef Eigen::Matrix<double, 8, 1> Mat81;
typedef Eigen::Matrix<double, 1, 8> Mat18;
typedef Eigen::Matrix<double, 9, 1> Mat91;
typedef Eigen::Matrix<double, 1, 9> Mat19;


typedef Eigen::Matrix<double, 8, 4> Mat84;
typedef Eigen::Matrix<double, 4, 8> Mat48;
typedef Eigen::Matrix<double, 4, 4> Mat44;


typedef Eigen::Matrix<float, MAX_RES_PER_POINT, 1> VecNRf;
typedef Eigen::Matrix<float, 12, 1> Vec12f;
typedef Eigen::Matrix<float, 1, 8> Mat18f;
typedef Eigen::Matrix<float, 1, 10> Mat110f;
typedef Eigen::Matrix<float, 6, 6> Mat66f;
typedef Eigen::Matrix<float, 8, 8> Mat88f;
typedef Eigen::Matrix<float, 8, 4> Mat84f;
typedef Eigen::Matrix<float, 8, 1> Vec8f;
typedef Eigen::Matrix<float, 10, 1> Vec10f;
typedef Eigen::Matrix<float, 6, 6> Mat66f;
typedef Eigen::Matrix<float, 4, 1> Vec4f;
typedef Eigen::Matrix<float, 4, 4> Mat44f;
typedef Eigen::Matrix<float, 3, 4> Mat34f;
typedef Eigen::Matrix<float, 11, 11> Mat1111f;
typedef Eigen::Matrix<float, 11, 1> Vec11f;
typedef Eigen::Matrix<float, 12, 12> Mat1212f;
typedef Eigen::Matrix<float, 12, 1> Vec12f;
typedef Eigen::Matrix<float, 13, 13> Mat1313f;
typedef Eigen::Matrix<float, 15, 15> Mat1515f;
typedef Eigen::Matrix<float, 10, 10> Mat1010f;
typedef Eigen::Matrix<float, 13, 1> Vec13f;
typedef Eigen::Matrix<float, 9, 9> Mat99f;
typedef Eigen::Matrix<float, 9, 1> Vec9f;

typedef Eigen::Matrix<float, 4, 2> Mat42f;
typedef Eigen::Matrix<float, 6, 2> Mat62f;
typedef Eigen::Matrix<float, 1, 2> Mat12f;

typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VecXf;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> MatXXf;


typedef Eigen::Matrix < double, 8 + CPARS + 1, 8 + CPARS + 1 > MatPCPC;
typedef Eigen::Matrix < float, 8 + CPARS + 1, 8 + CPARS + 1 > MatPCPCf;
typedef Eigen::Matrix < double, 8 + CPARS + 1, 1 > VecPC;
typedef Eigen::Matrix < float, 8 + CPARS + 1, 1 > VecPCf;

typedef Eigen::Matrix < double, 10 + CPARS + 1, 10 + CPARS + 1 > MatPCPC15;
typedef Eigen::Matrix < float, 10 + CPARS + 1, 10 + CPARS + 1 > MatPCPC15f;
typedef Eigen::Matrix < double, 10 + CPARS + 1, 1 > VecPC15;
typedef Eigen::Matrix < float, 10 + CPARS + 1, 1 > VecPC15f;

typedef Eigen::Matrix<float, 14, 14> Mat1414f;
typedef Eigen::Matrix<float, 14, 1> Vec14f;
typedef Eigen::Matrix<double, 14, 14> Mat1414;
typedef Eigen::Matrix<double, 14, 1> Vec14;


// I_frame = exp(a)*I_global + b. // I_global = exp(-a)*(I_frame - b).
//每个点像素的变换光度GAffine变换，即affine brightness transfer function 线性相应的逆函数
// transforms points from one frame to another.
struct AffLight
{
	AffLight(double a_, double b_) : a(a_), b(b_) {};
	AffLight() : a(0), b(0) {};

	// Affine Parameters:
	double a, b;	// I_frame = exp(a)*I_global + b. // I_global = exp(-a)*(I_frame - b).

	//用来计算两帧间的曝光差
	static Vec2 fromToVecExposure(float exposureF, float exposureT, AffLight g2F, AffLight g2T)
	{
		if (exposureF == 0 || exposureT == 0)
		{
			exposureT = exposureF = 1;
			//printf("got exposure value of 0! please choose the correct model.\n");
			//assert(setting_brightnessTransferFunc < 2);
		}
		//refToFh[0]=a＝e^(aj-ai)*tj*ti^(-1),两帧间的光度曝光变化
		double a = exp(g2T.a - g2F.a) * exposureT / exposureF;
		//refToFh[1]=b = 当前帧的b - refToFh[0]*当前帧的b
		double b = g2T.b - a * g2F.b;
		return Vec2(a, b);
	}

	Vec2 vec()
	{
		return Vec2(a, b);
	}
};

// DBoW vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;

}

