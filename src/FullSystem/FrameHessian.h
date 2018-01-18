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
#define MAX_ACTIVE_FRAMES 100


#include "util/globalCalib.h"
#include "vector"

#include <iostream>
#include <fstream>
#include <mutex>
#include <map>
#include "util/NumType.h"
#include "util/settings.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/FrameShell.h"
#include "util/ImageAndExposure.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/eigen.hpp>

namespace fdso
{


inline Vec2 affFromTo(Vec2 from, Vec2 to) // contains affine parameters as XtoWorld.
{
	return Vec2(from[0] / to[0], (from[1] - to[1]) / to[0]);
}


struct FrameHessian;
struct PointHessian;

class ImmaturePoint;
class FrameShell;

class EFFrame;
class EFPoint;

class Feature;

/**
 * 每一帧预计算的残差
 */
struct FrameFramePrecalc
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// static values
	// 状态量
	static int instanceCounter;
	//主导帧
	FrameHessian* host; // defines row
	//目标帧
	FrameHessian* target; // defines column

	// precalc values
	// 预计算的值
	Mat33f PRE_RTll;  //R
	Mat33f PRE_KRKiTll; //K*R*K'
	Mat33f PRE_RKiTll;  //R*K'
	Mat33f PRE_RTll_0;  //R evalPT

	Vec2f PRE_aff_mode; //主导帧和目标帧之间的光度ａ和ｂ变化
	float PRE_b0_mode;  //主导帧的b

	Vec3f PRE_tTll; //t
	Vec3f PRE_KtTll;  //R*t
	Vec3f PRE_tTll_0; //t PRE_RTll_0

	float distanceLL;


	inline ~FrameFramePrecalc() {}
	inline FrameFramePrecalc() {host = target = 0;}
	void set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib);
};

/**
 * 每一帧的信息
 */
struct FrameHessian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	//帧的能量函数
	EFFrame* efFrame;

	// constant info & pre-calculated values
	//DepthImageWrap* frame;
	//每一帧的信息
	FrameShell* shell;

	//梯度
	Eigen::Vector3f* dI;         // trace, fine tracking. Used for direction select (not for gradient histograms etc.)
	//用于跟踪，和初始化
	Eigen::Vector3f* dIp[PYR_LEVELS];  // coarse tracking / coarse initializer. NAN in [0] only.
	//用于像素的选择，用直方图，金字塔
	float* absSquaredGrad[PYR_LEVELS];  // only used for pixel select (histograms etc.). no NAN.

	//关键帧id
	int frameID;            // incremental ID for keyframes only!
	static int instanceCounter;
	//当前窗口中的帧id
	int idx;

	// Photometric Calibration Stuff
	// 光度矫正，根据跟踪残差动态设置
	float frameEnergyTH;  // set dynamically depending on tracking residual
	//即为参数t,用来表示曝光时间
	float ab_exposure;

	//是否边缘化
	bool flaggedForMarginalization;

	//有效点
	std::vector<PointHessian*> pointHessians;       // contains all ACTIVE points.
	//已边缘化的点,在flagPointsForRemoval中根据点的逆深度状态，插入
	std::vector<PointHessian*> pointHessiansMarginalized; // contains all MARGINALIZED points (= fully marginalized, usually because point went OOB.)
	//出界点/外点,在flagPointsForRemoval中插入
	std::vector<PointHessian*> pointHessiansOut;    // contains all OUTLIER points (= discarded.).

	//当前帧生成的点
	std::vector<ImmaturePoint*> immaturePoints;   // contains all OUTLIER points (= discarded.).

	cv::Mat image;
	// 金字塔，越往上越小，默认缩放倍数是2，因为2可以用SSE优化...虽然目前还没有用SSE
	std::vector<cv::Mat>  _pyramid;      // gray image pyramid, it must be CV_8U

	//特征点
	std::vector<Feature*> _features;

	DBoW2::BowVector _bow_vec;
	DBoW2::FeatureVector _feature_vec;

        // Variables used by the keyframe database
        //闭环时用到额
        long unsigned int mnLoopQuery = 0;
        int mnLoopWords = 0;
        float mLoopScore = 0;
        long unsigned int mnRelocQuery = 0;
        int mnRelocWords = 0;
        float mRelocScore = 0;

	// pose relative to keyframes in the window, stored as T_cur_ref
	// this will be changed by full system and loop closing, so we need a mutex
	std::map<FrameHessian*, SE3, std::less<FrameHessian*>, Eigen::aligned_allocator<SE3>> mPoseRel;
	std::mutex mMutexPoseRel;

	//零空间位姿
	Mat66 nullspaces_pose;
	//零空间的光度参数
	Mat42 nullspaces_affine;
	//零空间的尺度
	Vec6 nullspaces_scale;

	// variable info.
	// 相机在世界坐标系的位姿
	SE3 worldToCam_evalPT;
	//零状态
	Vec10 state_zero;
	//尺度状态，实际用的
	Vec10 state_scaled;
	//每次设置的状态，0-5为世界坐标系到相机坐标系的左扰动，6和7为光度参数
	Vec10 state;  // [0-5: worldToCam-leftEps. 6-7: a,b]
	Vec10 step;
	Vec10 step_backup;
	Vec10 state_backup;


	EIGEN_STRONG_INLINE const SE3 &get_worldToCam_evalPT() const {return worldToCam_evalPT;}
	EIGEN_STRONG_INLINE const Vec10 &get_state_zero() const {return state_zero;}
	EIGEN_STRONG_INLINE const Vec10 &get_state() const {return state;}
	EIGEN_STRONG_INLINE const Vec10 &get_state_scaled() const {return state_scaled;}
	EIGEN_STRONG_INLINE const Vec10 get_state_minus_stateZero() const {return get_state() - get_state_zero();}

	// precalc values
	// 预计算的值
	SE3 PRE_worldToCam;
	SE3 PRE_camToWorld;
	//目标帧预计算的值
	std::vector<FrameFramePrecalc, Eigen::aligned_allocator<FrameFramePrecalc>> targetPrecalc;
	//调试图像信息
	MinimalImageB3* debugImage;

	//获取世界坐标系到相机坐标系的左扰动
	inline Vec6 w2c_leftEps() const {return get_state_scaled().head<6>();}
	inline AffLight aff_g2l() const {return AffLight(get_state_scaled()[6], get_state_scaled()[7]);}
	inline AffLight aff_g2l_0() const {return AffLight(get_state_zero()[6] * SCALE_A, get_state_zero()[7] * SCALE_B);}

	/**
	 * [setStateZero description]
	 * @param state_zero [description]
	 * 设置零状态
	 */
	void setStateZero(Vec10 state_zero);
	/**
	 * [setState description]
	 * @param state [description]
	 * 设置状态
	 */
	inline void setState(Vec10 state)
	{
		//设置state
		this->state = state;
		//获取sstate_scaled从第0个开始的3个值
		//设置state_scaled，即设置左扰动w2c_leftEps
		//state与state_scaled差了SCALE
		state_scaled.segment<3>(0) = SCALE_XI_TRANS * state.segment<3>(0);
		state_scaled.segment<3>(3) = SCALE_XI_ROT * state.segment<3>(3);
		state_scaled[6] = SCALE_A * state[6];
		state_scaled[7] = SCALE_B * state[7];
		state_scaled[8] = SCALE_A * state[8];
		state_scaled[9] = SCALE_B * state[9];

		//左扰动的指数映射*worldToCam_evalPT
		PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
		PRE_camToWorld = PRE_worldToCam.inverse();
		//setCurrentNullspace();
	};

	/**
	 * [setStateScaled description]
	 * @param state_scaled [description]
	 * 设置状态的逆
	 */
	inline void setStateScaled(Vec10 state_scaled)
	{
		//设置state_scaled，即设置左扰动w2c_leftEps
		this->state_scaled = state_scaled;

		//设置state
		//state与state_scaled差了SCALE INVERSE
		state.segment<3>(0) = SCALE_XI_TRANS_INVERSE * state_scaled.segment<3>(0);
		state.segment<3>(3) = SCALE_XI_ROT_INVERSE * state_scaled.segment<3>(3);
		state[6] = SCALE_A_INVERSE * state_scaled[6];
		state[7] = SCALE_B_INVERSE * state_scaled[7];
		state[8] = SCALE_A_INVERSE * state_scaled[8];
		state[9] = SCALE_B_INVERSE * state_scaled[9];

		//左扰动的指数映射*worldToCam_evalPT
		PRE_worldToCam = SE3::exp(w2c_leftEps()) * get_worldToCam_evalPT();
		PRE_camToWorld = PRE_worldToCam.inverse();
		//setCurrentNullspace();
	};

	/**
	 * [setEvalPT description]
	 * @param worldToCam_evalPT [description]
	 * @param state             [description]
	 */
	inline void setEvalPT(SE3 worldToCam_evalPT, Vec10 state)
	{
		this->worldToCam_evalPT = worldToCam_evalPT;
		setState(state);
		setStateZero(state);
	};

	/**
	 * [setEvalPT_scaled description]
	 * @param worldToCam_evalPT [description]
	 * @param aff_g2l           [description]
	 */
	inline void setEvalPT_scaled(SE3 worldToCam_evalPT, AffLight aff_g2l)
	{
		Vec10 initial_state = Vec10::Zero();
		initial_state[6] = aff_g2l.a;
		initial_state[7] = aff_g2l.b;
		this->worldToCam_evalPT = worldToCam_evalPT;
		setStateScaled(initial_state);
		setStateZero(this->get_state());
	};

	void release();

	/**
	 * 消除
	 */
	inline ~FrameHessian()
	{
		assert(efFrame == 0);
		release(); instanceCounter--;
		for (int i = 0; i < pyrLevelsUsed; i++)
		{
			delete[] dIp[i];
			delete[]  absSquaredGrad[i];

		}
		if (debugImage != 0) delete debugImage;
	};

	/**
	 * [FrameHessian description]
	 * @return [description]
	 * 初始化
	 */
	inline FrameHessian()
	{
		instanceCounter++;
		flaggedForMarginalization = false;
		frameID = -1;
		efFrame = 0;
		frameEnergyTH = 8 * 8 * patternNum;
		debugImage = 0;
	};

	/**
	 * [makeImages description]
	 * @param color          [description]
	 * @param overexposedMap [description]
	 * @param HCalib         [description]
	 * 构建帧的梯度图，构建帧的雅克比和Hessian矩阵
	 */
	void makeImages(ImageAndExposure* imageE, CalibHessian* HCalib);

	// 将备选点的描述转换成 bow
	void ComputeBoW(ORBVocabulary* _vocab);

	set<FrameHessian*> GetConnectedKeyFrames();

	void CleanAllFeatures()
	{
		for ( size_t i = 0; i < _features.size(); i++ )
		{
			delete _features[i];
		}
		_features.clear();
	}

	inline bool InFrame( const Eigen::Vector2d& pixel, const int& boarder = 10 ) const
	{
		return pixel[0] >= boarder && pixel[0] < image.cols - boarder
		       && pixel[1] >= boarder && pixel[1] < image.rows - boarder;
	}

	inline bool InFrame( const cv::Point2f& pixel, const int& boarder = 10 ) const
	{
		return pixel.x >= boarder && pixel.x < image.cols - boarder
		       && pixel.y >= boarder && pixel.y < image.rows - boarder;
	}

// 带level的查询
	inline bool InFrame( const Eigen::Vector2d& pixel, const int& boarder, const int& level ) const
	{
		return pixel[0] / (1 << level) >= boarder && pixel[0] / (1 << level) < image.cols - boarder
		       && pixel[1] / (1 << level) >= boarder && pixel[1] / (1 << level) < image.rows - boarder;
	}

	/**
	 * [getPrior description]
	 * @return [description]
	 * 获得先验的
	 */
	inline Vec10 getPrior()
	{
		Vec10 p =  Vec10::Zero();
		//第一个关键帧
		if (frameID == 0)
		{
			//前三位位置，1e10
			p.head<3>() = Vec3::Constant(setting_initialTransPrior);
			//后三位旋转，1e11
			p.segment<3>(3) = Vec3::Constant(setting_initialRotPrior);
			if (setting_solverMode & SOLVER_REMOVE_POSEPRIOR)
				p.head<6>().setZero();

			//初始的a和b为1e14
			p[6] = setting_initialAffAPrior;
			p[7] = setting_initialAffBPrior;
		}
		else
		{
			if (setting_affineOptModeA < 0)
				p[6] = setting_initialAffAPrior;
			else
				p[6] = setting_affineOptModeA;

			if (setting_affineOptModeB < 0)
				p[7] = setting_initialAffBPrior;
			else
				p[7] = setting_affineOptModeB;
		}
		p[8] = setting_initialAffAPrior;
		p[9] = setting_initialAffBPrior;
		return p;
	}

	/**
	 * [getPriorZero description]
	 * @return [description]
	 * 获得零Hessian
	 */
	inline Vec10 getPriorZero()
	{
		return Vec10::Zero();
	}

};

/**
 * Compare frame ID, used to get a sorted map or set of frames
 * 比较帧的id
 */
class CmpFrameID {
public:
	inline bool operator()(const FrameHessian* f1, const FrameHessian* f2) {
		return f1->shell->id < f2->shell->id;
	}
};

/**
 * Compare frame by Keyframe ID, used to get a sorted keyframe map or set.
 * 比较帧的关键帧id
 */
class CmpFrameKFID {
public:
	inline bool operator()(const FrameHessian* f1, const FrameHessian* f2) {
		return f1->frameID < f2->frameID;
	}
};

}

