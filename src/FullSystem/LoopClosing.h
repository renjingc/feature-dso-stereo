#pragma once

#include "util/NumType.h"
#include "FullSystem/Map.h"
#include "FullSystem/FeatureMatcher.h"

#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"

#include "FullSystem/Frame.h"
#include "FullSystem/PointHessian.h"
#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/CalibHessian.h"

#include <list>
#include <queue>
#include <mutex>

using namespace std;

using fdso::CalibHessian;

namespace fdso {
class FullSystem;
// namespace IOWrap
// {
//     class Output3DWrapper;
// }

/**
 * keyframe database, use to give loop candidates
 */
class KeyFrameDatabase {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    KeyFrameDatabase(std::shared_ptr<ORBVocabulary> voc);

    void add(Frame* pKF);

    void erase(Frame* pKF);

    void clear();

    // Loop Detection
    std::vector<Frame*> DetectLoopCandidates(Frame* pKF, float minScore);

private:
    std::shared_ptr<ORBVocabulary> mpVoc;

    std::vector<list<Frame*>> mvInvertedFile; ///< 倒排索引，mvInvertedFile[i]表示包含了第i个word id的所有关键帧

    // Mutex
    std::mutex mMutex;
};

/**
 * Loop closing thread, also used for correcting loops
 *
 * loop closing is running in a single thread, receiving new keyframes from the front end. It will seprate all keyframes into several "consistent groups", and if the newest keyframe seems to be consistent with a previous group, we say a loop closure is detected. And once we find a loop, we will check the sim3 and correct it using a global bundle adjustment.
 */
class LoopClosing
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // Consistent group, the first is a group of keyframes that are considered as consistent, and the second is how many times they are detected
    typedef pair<set<Frame*>, int> ConsistentGroup;

    LoopClosing(FullSystem* fullsystem);

    void insertKeyFrame(Frame* frame);

    /**
     * detect loop candidates from the keyframe database
     * @param frame
     * @return true if there is at least one loop candidate
     */
    bool DetectLoop(Frame* frame);

    /**
     * compute RANSAC pnp in loop frames
     * however this function will not try to optimize the pose graph, which will be done in full system
     * @return true if find enough inliers
     */
    bool CorrectLoop(CalibHessian* Hcalib);

    void run();

    /**
     * set main loop to finish
     * @param finish
     */
    void setFinish(bool finish = true)
    {
        mbNeedFinish = finish;
    }

private:

    // data
    shared_ptr<Map> mpGlobalMap = nullptr;  // global map
    KeyFrameDatabase* mpKeyFrameDB = nullptr;
    std::shared_ptr<ORBVocabulary> mpVoc;

    std::vector<ConsistentGroup> mvConsistentGroups;    // many groups
    std::vector<Frame*> mvpEnoughConsistentCandidates;  // loop candidate frames compared with the newest one.

    vector<Frame*> mvAllKF;
    Frame* mpCurrentKF = nullptr;

    // loop kf queue
    deque<Frame*> mvKFQueue;
    mutex mutexKFQueue;

    bool mbFinished = false;
    CalibHessian* mpHcalib = nullptr;
    bool mbNeedFinish = false;

    thread mainLoop;

    // parameters
    float mnCovisibilityConsistencyTh = 3;   // how many times a keyframe is checked as consistent
};
}