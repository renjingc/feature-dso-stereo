#include "LoopClosing.h"
#include "util/GlobalCalib.h"
#include "util/settings.h"
#include "util/NumType.h"
#include "FullSystem.h"

#include <opencv2/calib3d/calib3d.hpp>

namespace fdso {

KeyFrameDatabase::KeyFrameDatabase(const ORBVocabulary* voc)
{
    mpVoc = voc;
    mvInvertedFile.resize(voc->size());
}

void KeyFrameDatabase::add(FrameHessian* pKF)
{
    unique_lock<mutex> lock(mMutex);
    for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end(); vit != vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}

void KeyFrameDatabase::erase(shared_ptr<Frame> pKF)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end();
            vit != vend; vit++)
    {
        // List of keyframes that share the word
        auto &lKFs = mvInvertedFile[vit->first];

        for (auto lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++)
        {
            if (pKF == *lit) {
                lKFs.erase(lit);
                break;
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());
}

std::vector<shared_ptr<Frame>> KeyFrameDatabase::DetectLoopCandidates(shared_ptr<Frame> &pKF, float minScore)
{
    set<shared_ptr<Frame>> spConnectedKeyFrames = pKF->GetConnectedKeyFrames(); // connected kfs

    // find keyframes sharing same words
    list<shared_ptr<Frame>> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    {
        unique_lock<mutex> lock(mMutex);

        for (DBoW2::BowVector::const_iterator vit = pKF->mBowVec.begin(), vend = pKF->mBowVec.end();
                vit != vend; vit++)
        {
            auto &lKFs = mvInvertedFile[vit->first];

            for (auto lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++) {
                shared_ptr<Frame> pKFi = *lit;

                if (pKFi->mnLoopQuery != pKF->mKFId) {
                    pKFi->mnLoopWords = 0;

                    if (!spConnectedKeyFrames.count(pKFi)) {
                        // not contained in window
                        pKFi->mnLoopQuery = pKF->mKFId;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnLoopWords++;
            }
        }
    }

    if (lKFsSharingWords.empty())
        return vector<shared_ptr<Frame>>();

    list<pair<float, shared_ptr<Frame>>> lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    int maxCommonWords = (*max_element(lKFsSharingWords.begin(), lKFsSharingWords.end(),
    [](const shared_ptr<Frame> &fr1, const shared_ptr<Frame> &fr2) {
        return fr1->mnLoopWords < fr2->mnLoopWords;
    }))->mnLoopWords;

    int minCommonWords = maxCommonWords * 0.8f;
    int nscores = 0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for (auto lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end();
            lit != lend; lit++) {
        auto pKFi = *lit;

        if (pKFi->mnLoopWords > minCommonWords) {
            nscores++;
            float si = mpVoc->score(pKF->mBowVec, pKFi->mBowVec);
            pKFi->mLoopScore = si;
            if (si >= minScore)
                lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
        return vector<shared_ptr<Frame>>();

    list<pair<float, shared_ptr<Frame>>> lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    for (auto it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++) {
        shared_ptr<Frame> pKFi = it->second;
        set<shared_ptr<Frame>> vpNeighs = pKFi->GetConnectedKeyFrames();

        float bestScore = it->first;
        float accScore = it->first;

        shared_ptr<Frame> pBestKF = pKFi;
        for (auto vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++) {
            shared_ptr<Frame> pKF2 = *vit;

            if (pKF2->mnLoopQuery == pKF->mKFId && pKF2->mnLoopWords > minCommonWords) {
                accScore += pKF2->mLoopScore;
                if (pKF2->mLoopScore > bestScore) {
                    pBestKF = pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore, pBestKF));
        if (accScore > bestAccScore)
            bestAccScore = accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f * bestAccScore;

    set<shared_ptr<Frame>> spAlreadyAddedKF;
    vector<shared_ptr<Frame>> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for (auto it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end();
            it != itend; it++) {
        if (it->first > minScoreToRetain) {
            shared_ptr<Frame> pKFi = it->second;
            if (!spAlreadyAddedKF.count(pKFi)) {
                // LOG(INFO) << "add candidate " << pKFi->mKFId << ", sim score: " << it->first << endl;
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }
    return vpLoopCandidates;
}


// -----------------------------------------------------------
LoopClosing::LoopClosing(FullSystem *fullsystem) :
    mpKeyFrameDB(new KeyFrameDatabase(fullsystem->vocab)), mpVoc(fullsystem->vocab),
    mpGlobalMap(fullsystem->globalMap), mpHcalib(fullsystem->Hcalib->mpCH)
{
    mainLoop = thread(&LoopClosing::run, this);
}

void LoopClosing::insertKeyFrame(shared_ptr<Frame> &frame)
{
    unique_lock<mutex> lock(mutexKFQueue);
    mvKFQueue.push_back(frame);
}

void LoopClosing::run()
{
    mbFinished = false;

    while (1)
    {
        {
            // get the oldest one
            unique_lock<mutex> lock(mutexKFQueue);
            if (mvKFQueue.empty())
            {
                lock.unlock();
                usleep(5000);
                continue;
            }
            mpCurrentKF = mvKFQueue.front();
            mvKFQueue.pop_front();
            //mvAllKF.push_back(mpCurrentKF);
        }

        mpCurrentKF->ComputeBoW(mpVoc);
        if (DetectLoop(mpCurrentKF))
        {
            if (mpGlobalMap->idle() && CorrectLoop(mpHcalib))
            {
                // start a pose graph optimization
                LOG(INFO) << "call global pose graph!" << endl;
                mpGlobalMap->optimizeALLKFs();
            }
        }

        if (mbNeedFinish)
            break;
        usleep(5000);
    }

    mbFinished = true;
}

bool LoopClosing::DetectLoop(shared_ptr<Frame> &frame)
{
    // compute minimum similarity score
    auto connectKFs = frame->GetConnectedKeyFrames();
    float minScore = 1.0;
    for (auto &fr : connectKFs)
    {
        if (fr == frame)
            continue;
        float score = mpVoc->score(fr->mBowVec, frame->mBowVec);
        if (score > 0 && score < minScore)
            minScore = score;
    }

    // query the database imposing the minimum score
    vector<shared_ptr<Frame>> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(frame, minScore);

    if (vpCandidateKFs.empty())
    {
        mpKeyFrameDB->add(frame);
        mvConsistentGroups.clear();
        LOG(INFO) << "No candidates" << endl;
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it

    LOG(INFO) << "candidates: " << vpCandidateKFs.size() << endl;
    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(), false);

    for (auto &pCandidateKF : vpCandidateKFs)
    {
        // check consistency
        set<shared_ptr<Frame>> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;

        for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG < iendG; iG++)
        {
            set<shared_ptr<Frame>> sPreviousGroup = mvConsistentGroups[iG].first;
            bool bConsistent = false;

            for (auto sit = spCandidateGroup.begin(), send = spCandidateGroup.end(); sit != send; sit++)
            {
                if (sPreviousGroup.count(*sit))
                {
                    // we have seen it before
                    bConsistent = true;
                    bConsistentForSomeGroup = true;
                    break;
                }
            }

            if (bConsistent)
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                if (!vbConsistentGroup[iG])
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup, nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG] = true; //this avoid to include the same group more than once
                }
                if (nCurrentConsistency >= mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    // enough candidate
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent = true; //this avoid to insert the same candidate more than once
                }
            }
        }

        if (!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup, 0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;

    // Add Current Keyframe to database
    mpKeyFrameDB->add(frame);

    if (mvpEnoughConsistentCandidates.empty())
    {
        return false;
    }
    else
    {
        for (auto &fr : mvpEnoughConsistentCandidates)
        {
            LOG(INFO) << "candidate: " << fr->mKFId << endl;
        }
        return true;
    }
}

bool LoopClosing::CorrectLoop(shared_ptr<CalibHessian> Hcalib)
{
    LOG(INFO) << "Found " << mvpEnoughConsistentCandidates.size() << " with current kf" << endl;
    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    FeatureMatcher matcher(0.75, true);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    bool success = false;

    int nCandidates = 0; //candidates with enough matches

    // intrinsics
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = Hcalib->fxl();
    K.at<float>(1, 1) = Hcalib->fyl();
    K.at<float>(0, 2) = Hcalib->cxl();
    K.at<float>(1, 2) = Hcalib->cyl();

    for (int i = 0; i < nInitialCandidates; i++)
    {
        shared_ptr<Frame> pKF = mvpEnoughConsistentCandidates[i];
        LOG(INFO) << "try " << mpCurrentKF->mKFId << " with " << pKF->mKFId << endl;

        auto connectedKFs = pKF->GetConnectedKeyFrames();
        vector<Match> matches;
        int nmatches = matcher.SearchByBoW(mpCurrentKF, pKF, matches);

        if (nmatches < 10)
        {
            LOG(INFO) << "no enough matches: " << nmatches << endl;
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            LOG(INFO) << "let's try opencv's solve pnp ransac first " << std::endl;
            // well let's try opencv's solve pnp ransac first
            // TODO sim3 maybe better, but DSO's scale is not likely to drift?
            vector<cv::Point3f> p3d;
            vector<cv::Point2f> p2d;
            cv::Mat inliers;
            vector<int> matchIdx;

            for (size_t k = 0; k < matches.size(); k++)
            {
                auto &m = matches[k];
                shared_ptr<Feature> &featKF = pKF->mFeatures[m.index2];
                shared_ptr<Feature> &featCurrent = mpCurrentKF->mFeatures[m.index1];

                if (featKF->mStatus == Feature::FeatureStatus::VALID &&
                        featKF->mpPoint->mStatus != Point::PointStatus::OUTLIER)
                {
                    // there should be a 3d point
                    shared_ptr<Point> &pt = featKF->mpPoint;
                    pt->ComputeWorldPos();
                    cv::Point3f pt3d(pt->mWorldPos[0], pt->mWorldPos[1], pt->mWorldPos[2]);
                    p3d.push_back(pt3d);
                    p2d.push_back(cv::Point2f(featCurrent->mfUV[0], featCurrent->mfUV[1]));
                    matchIdx.push_back(k);
                }
            }

            if (p3d.size() < 10)
            {
                continue;   // inlier not enough, abort it
            }

            cv::Mat R, t;
            cv::solvePnPRansac(p3d, p2d, K, cv::Mat(), R, t, false, 100, 8.0, 100, inliers);

            int cntInliers = 0;

            for (size_t k = 0; k < inliers.rows; k++)
            {
                cntInliers++;
            }

            if (cntInliers < 5)
                return false;

            LOG(INFO) << "Loop detected from kf " << mpCurrentKF->mKFId << " to " << pKF->mKFId
                      << ", inlier matches: " << cntInliers << endl;

            // and then test with the estimated Tcw
            SE3 TcwEsti(
                SO3::exp(Vec3(R.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(2, 0))),
                Vec3(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0)));


            for (auto &kfn : pKF->mPoseRel)
            {
                SE3 TCurKF = TcwEsti * kfn.first->mTcwOpti.inverse();
                unique_lock<mutex> lock(mpCurrentKF->mMutexPoseRel);
                mpCurrentKF->mPoseRel[kfn.first] = TCurKF;   // add an pose graph edge
            }

            {
                SE3 TCurRef = TcwEsti * pKF->mTcwOpti.inverse();
                unique_lock<mutex> lock(mpCurrentKF->mMutexPoseRel);
                mpCurrentKF->mPoseRel[pKF] = TCurRef;   // and an pose graph edge
            }

            SE3 delta = TcwEsti * mpCurrentKF->mTcwOpti.inverse();
            double dd = delta.log().norm();

            if (dd > 0.05)   // if we find a large error, start a pose graph optimization
                success = true;
        }
        nCandidates++;
    }
    return success;
}

}
