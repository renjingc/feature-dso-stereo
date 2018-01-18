#pragma once

#include "util/NumType.h"
#include "FullSystem/Map.h"
#include "FullSystem/FeatureMatcher.h"

#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"

#include "FullSystem/FrameHessian.h"
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

    /**
     * keyframe database, use to give loop candidates
     */
    class KeyFrameDatabase {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        KeyFrameDatabase(ORBVocabulary* voc);

        void add(FrameHessian* pKF);

        void erase(FrameHessian* pKF);

        void clear();

        // Loop Detection
        std::vector<FrameHessian*> DetectLoopCandidates(FrameHessian* pKF, float minScore);

    private:
        ORBVocabulary* mpVoc;

        std::vector<list<FrameHessian*>> mvInvertedFile; ///< 倒排索引，mvInvertedFile[i]表示包含了第i个word id的所有关键帧

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
        typedef pair<set<FrameHessian*>, int> ConsistentGroup;

        LoopClosing(FullSystem* fullSystem);

        void insertKeyFrame(FrameHessian* frame);

        /**
         * detect loop candidates from the keyframe database
         * @param frame
         * @return true if there is at least one loop candidate
         */
        bool DetectLoop(FrameHessian* frame);

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
        ORBVocabulary* mpVoc;

        std::vector<ConsistentGroup> mvConsistentGroups;    // many groups
        std::vector<FrameHessian*> mvpEnoughConsistentCandidates;  // loop candidate frames compared with the newest one.

        vector<FrameHessian*> mvAllKF;
        FrameHessian* mpCurrentKF = nullptr;

        // loop kf queue
        deque<FrameHessian*> mvKFQueue;
        mutex mutexKFQueue;

        bool mbFinished = false;
        CalibHessian* mpHcalib = nullptr;
        bool mbNeedFinish = false;

        thread mainLoop;

        // parameters
        float mnCovisibilityConsistencyTh = 3;   // how many times a keyframe is checked as consistent
    };
}