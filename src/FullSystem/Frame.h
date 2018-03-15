#pragma once


#include "util/NumType.h"
#include "FullSystem/FrameHessian.h"
#include "FullSystem/PointHessian.h"
#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/CalibHessian.h"
#include "FullSystem/FrameShell.h"
#include "FullSystem/FeatureDetector.h"

#include <set>
#include <thread>
#include <mutex>


using namespace std;

namespace fdso {
class FrameHessian;
class Feature;

class Frame
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Frame()
    {
        camToWorld = SE3();
        camToWorldOpti = SE3();
        frameID = -1;
        id=-1;
    }
    Frame(FrameHessian* frame);

    inline ~Frame()
    {
        release();
    }

    void release();

    bool update;
    bool lastUpdate;

    int id;
    int frameID;
    double timestamp;       // timestamp passed into DSO.
    // constantly adapted.
    //相对于世界坐标系的变换
    SE3 camToWorld;
    SE3 camToWorldOpti;

    // pose relative to keyframes in the window, stored as T_cur_ref
    // this will be changed by full system and loop closing, so we need a mutex
    std::map<Frame*, SE3, std::less<Frame*>, Eigen::aligned_allocator<SE3>> mPoseRel;
    std::mutex mMutexPoseRel;

    set<Frame*> GetConnectedKeyFrames();


    //特征点
    std::vector<Feature*> _features;

    DBoW2::BowVector _bow_vec;
    DBoW2::FeatureVector _feature_vec;

    //闭环时用到额
    long unsigned int mnLoopQuery = 0;
    int mnLoopWords = 0;
    float mLoopScore = 0;
    long unsigned int mnRelocQuery = 0;
    int mnRelocWords = 0;
    float mRelocScore = 0;

};


/**
 * Compare frame ID, used to get a sorted map or set of frames
 * 比较帧的id
 */
class CmpFrameID {
public:
    inline bool operator()(const Frame* f1, const Frame* f2) {
        if(f1->id < f2->id)
            return true;
        else
            return false;
    }
};

/**
 * Compare frame by Keyframe ID, used to get a sorted keyframe map or set.
 * 比较帧的关键帧id
 */
class CmpFrameKFID {
public:
    inline bool operator()(const Frame* f1, const Frame* f2) {
        if(f1->frameID < f2->frameID)
            return true;
        else
            return false;
    }
};

}
