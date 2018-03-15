#include "FullSystem/LoopClosing.h"
#include "util/globalCalib.h"
#include "util/settings.h"
#include "util/NumType.h"
#include "FullSystem/FullSystem.h"

#include <opencv2/calib3d/calib3d.hpp>

namespace fdso {



KeyFrameDatabase::KeyFrameDatabase(std::shared_ptr<ORBVocabulary> voc)
{
    mpVoc = voc;
    mvInvertedFile.resize(voc->size());
}

void KeyFrameDatabase::add(Frame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutex);
    for (DBoW2::BowVector::const_iterator vit = pKF->_bow_vec.begin(), vend = pKF->_bow_vec.end(); vit != vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF);
}

void KeyFrameDatabase::erase(Frame* pKF)
{
    std::unique_lock<std::mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for (DBoW2::BowVector::const_iterator vit = pKF->_bow_vec.begin(), vend = pKF->_bow_vec.end();
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

std::vector<Frame*> KeyFrameDatabase::DetectLoopCandidates(Frame* pKF, float minScore)
{
    //获取关联的帧
    set<Frame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames(); // connected kfs

    // find keyframes sharing same words
    //发现相似的帧
    list<Frame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    // 搜索所有关键帧与当前帧的词包，分享
    // 丢弃连接到查询关键帧关键帧
    {
        std::unique_lock<std::mutex> lock(mMutex);

        for (DBoW2::BowVector::const_iterator vit = pKF->_bow_vec.begin(), vend = pKF->_bow_vec.end();
                vit != vend; vit++)
        {
            auto &lKFs = mvInvertedFile[vit->first];

            for (auto lit = lKFs.begin(), lend = lKFs.end(); lit != lend; lit++) {
                Frame* pKFi = *lit;

                if (pKFi->mnLoopQuery != pKF->frameID) {
                    pKFi->mnLoopWords = 0;

                    if (!spConnectedKeyFrames.count(pKFi)) {
                        // not contained in window
                        pKFi->mnLoopQuery = pKF->frameID;
                        lKFsSharingWords.push_back(pKFi);
                    }
                }
                pKFi->mnLoopWords++;
            }
        }
    }

    if (lKFsSharingWords.empty())
        return vector<Frame*>();

    list<pair<float, Frame*>> lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    int maxCommonWords = (*max_element(lKFsSharingWords.begin(), lKFsSharingWords.end(),
    [](const Frame * fr1, const Frame * fr2) {
        return fr1->mnLoopWords < fr2->mnLoopWords;
    }))->mnLoopWords;

    int minCommonWords = maxCommonWords * 0.8f;
    int nscores = 0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for (auto lit = lKFsSharingWords.begin(), lend = lKFsSharingWords.end();
            lit != lend; lit++) {
        auto pKFi = *lit;

        if (pKFi->mnLoopWords > minCommonWords)
        {
            nscores++;
            float si = mpVoc->score(pKF->_bow_vec, pKFi->_bow_vec);
            pKFi->mLoopScore = si;
            if (si >= minScore)
                lScoreAndMatch.push_back(make_pair(si, pKFi));
        }
    }

    if (lScoreAndMatch.empty())
        return vector<Frame*>();

    list<pair<float, Frame*>> lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    for (auto it = lScoreAndMatch.begin(), itend = lScoreAndMatch.end(); it != itend; it++)
    {
        Frame* pKFi = it->second;
        set<Frame*> vpNeighs = pKFi->GetConnectedKeyFrames();

        float bestScore = it->first;
        float accScore = it->first;

        Frame* pBestKF = pKFi;
        for (auto vit = vpNeighs.begin(), vend = vpNeighs.end(); vit != vend; vit++) {
            Frame* pKF2 = *vit;

            if (pKF2->mnLoopQuery == pKF->frameID && pKF2->mnLoopWords > minCommonWords) {
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

    set<Frame*> spAlreadyAddedKF;
    vector<Frame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for (auto it = lAccScoreAndMatch.begin(), itend = lAccScoreAndMatch.end();
            it != itend; it++) {
        if (it->first > minScoreToRetain)
        {
            Frame* pKFi = it->second;
            if (!spAlreadyAddedKF.count(pKFi))
            {
                // LOG(INFO) << "add candidate " << pKFi->frameID << ", sim score: " << it->first << endl;
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpLoopCandidates;
}


// -----------------------------------------------------------
LoopClosing::LoopClosing(FullSystem *fullsystem) :
    mpKeyFrameDB(new KeyFrameDatabase(fullsystem->_vocab)), mpVoc(fullsystem->_vocab),
    mpGlobalMap(fullsystem->globalMap), mpHcalib(&fullsystem->Hcalib)
{
    mainLoop = thread(&LoopClosing::run, this);
}

void LoopClosing::insertKeyFrame(Frame* frame)
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

        //mpCurrentKF->ComputeBoW(mpVoc);

        if (openLoop)
        {
            if (DetectLoop(mpCurrentKF))
            {
                if (mpGlobalMap->idle() && CorrectLoop(mpHcalib))
                {
                    // start a pose graph optimization
                    LOG(INFO) << "call global pose graph!" << endl;
                    mpGlobalMap->optimizeALLKFs();
                }
            }
        }

        if (mbNeedFinish)
            break;
        usleep(5000);

        // boost::timer t;
        // // //发布关键帧和当前窗口中帧的关联
        mpGlobalMap->outputWrapper[0]->publishKeyframesOpt(mpGlobalMap->frameList, false, mpHcalib);

        // std::cout << "publishKeyframesOpt t: " << t.elapsed() << std::endl;
    }

    mbFinished = true;
}

bool LoopClosing::DetectLoop(Frame* frame)
{
    // compute minimum similarity score
    //获取当前帧所有关联的帧
    auto connectKFs = frame->GetConnectedKeyFrames();
    //最小得分
    float minScore = 1.0;
    //所有的连接
    for (auto &fr : connectKFs)
    {
        //不是当前帧
        if (fr == frame)
            continue;
        //得到两帧的词包得分
        float score = mpVoc->score(fr->_bow_vec, frame->_bow_vec);
        //获取最低得分
        if (score > 0 && score < minScore)
            minScore = score;
    }

    // query the database imposing the minimum score
    //在数据库里
    vector<Frame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(frame, 2 * minScore);

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
        set<Frame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;

        for (size_t iG = 0, iendG = mvConsistentGroups.size(); iG < iendG; iG++)
        {
            set<Frame*> sPreviousGroup = mvConsistentGroups[iG].first;
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
            LOG(INFO) << "candidate: " << fr->frameID << endl;
        }
        return true;
    }
}

bool LoopClosing::CorrectLoop(CalibHessian* Hcalib)
{
    LOG(INFO) << "Found " << mvpEnoughConsistentCandidates.size() << " with current kf" << endl;
    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    FeatureMatcher matcher(65, 100, 30, 100, 0.7);

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
        Frame* pKF = mvpEnoughConsistentCandidates[i];
        LOG(INFO) << "try " << mpCurrentKF->frameID << " with " << pKF->frameID << endl;

        auto connectedKFs = pKF->GetConnectedKeyFrames();
        vector<cv::DMatch> matches, goodMatches;
        int nmatches = matcher.SearchByBoW(mpCurrentKF, pKF, matches);
        matcher.checkUVDistance(mpCurrentKF, pKF, matches, goodMatches);

        if (goodMatches.size() < 10)
        {
            LOG(INFO) << "no enough matches: " << goodMatches.size() << endl;
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
            //matcher.showMatch(mpCurrentKF,pKF,goodMatches);
            LOG(INFO) << "let's try opencv's solve pnp ransac first " << std::endl;
            // well let's try opencv's solve pnp ransac first
            // TODO sim3 maybe better, but DSO's scale is not likely to drift?
            vector<cv::Point3f> p3d;
            vector<cv::Point2f> p2d;
            cv::Mat inliers;
            vector<int> matchIdx;

            // LOG(INFO)<<"goodMatches size(): "<<goodMatches.size()<<std::endl;
            for (size_t k = 0; k < goodMatches.size(); k++)
            {
                auto &m = goodMatches[k];

                Feature* featKF = pKF->_features[m.trainIdx];
                Feature* featCurrent = mpCurrentKF->_features[m.queryIdx];

                if (featKF->_status == Feature::ACTIVE_IDEPTH)
                {
                    // there should be a 3d point
                    // PointHessian* pt = featKF->mPH;
                    //LOG(INFO) << "pKF pt3d (u,v,iepth): "<<" "<<pt->u<<" "<<pt->v<<" "<<pt->idepth << std::endl;
                    Vec3 mWorldPos;
                    featKF->ComputeWorldPos(mWorldPos);
                    //LOG(INFO) << "pKF pt3d: "<<" "<<pt->mWorldPos[0]<<" "<<pt->mWorldPos[1]<<" "<<pt->mWorldPos[2] << std::endl;
                    cv::Point3f pt3d(mWorldPos[0], mWorldPos[1], mWorldPos[2]);
                    p3d.push_back(pt3d);
                    //LOG(INFO) << "mpCurrentKF p2d: "<<" "<<featCurrent->_pixel[0]<<" "<<featCurrent->_pixel[1]<< std::endl;
                    p2d.push_back(cv::Point2f(featCurrent->_pixel[0], featCurrent->_pixel[1]));
                    matchIdx.push_back(k);
                }
            }

            if (p3d.size() < 10)
            {
                LOG(INFO) << "pKF p3d size() < 10 " << std::endl;
                continue;   // inlier not enough, abort it
            }

            LOG(INFO) << "solvePnPRansac size(): " << p3d.size() << " " << p2d.size() << std::endl;
            cv::Mat R, t;
            //cv::solvePnPRansac(p3d, p2d, K, cv::Mat(), R, t, false, 100, 8.0, 100, inliers);
            cv::solvePnPRansac(p3d, p2d, K, cv::Mat(), R, t, false, 100, 3.0, 0.99, inliers);

            int cntInliers = 0;

            for (size_t k = 0; k < inliers.rows; k++)
            {
                cntInliers++;
            }

            if (cntInliers < 10)
                return false;

            // Eigen::Matrix<double, 3, 1> mpCurrentKF_T = mpCurrentKF->camToWorldOpti.translation().transpose();
            // Eigen::Matrix<double, 3, 1> pKF_T = pKF->camToWorldOpti.translation().transpose();
            // std::cout<<"mpCurrentKF "<<mpCurrentKF_T(0,0)<<" "<<mpCurrentKF_T(1,0)<<" "<<mpCurrentKF_T(2,0)<<endl;
            // std::cout<<"pKF "<<pKF_T(0,0)<<" "<<pKF_T(1,0)<<" "<<pKF_T(2,0)<<endl;

            // std::cout<<"relative t r"<<t<<endl<<R<<endl;

            // and then test with the estimated Tcw
            SE3 TcwEsti(
                SO3::exp(Vec3(R.at<double>(0, 0), R.at<double>(1, 0), R.at<double>(2, 0))),
                Vec3(t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0)));


            for (auto &kfn : pKF->mPoseRel)
            {
                SE3 TCurKF = TcwEsti * kfn.first->camToWorldOpti;
                unique_lock<mutex> lock(mpCurrentKF->mMutexPoseRel);
                mpCurrentKF->mPoseRel[kfn.first] = TCurKF;   // add an pose graph edge
            }

            {
                SE3 TCurRef = TcwEsti * pKF->camToWorldOpti;
                unique_lock<mutex> lock(mpCurrentKF->mMutexPoseRel);
                mpCurrentKF->mPoseRel[pKF] = TCurRef;   // and an pose graph edge
            }

            SE3 delta = TcwEsti * mpCurrentKF->camToWorldOpti;
            double dd = delta.log().norm();

            if (dd > 0.1 && (mpCurrentKF->frameID - pKF->frameID) > 50) // if we find a large error, start a pose graph optimization
            {
                //mpCurrentKF->camToWorldOpti = TcwEsti;
                //mpCurrentKF->camToWorld = TcwEsti;
                std::cout << "Loop detected from kf " << mpCurrentKF->frameID << " to " << pKF->frameID
                          << ", inlier matches: " << cntInliers << endl;

                success = true;
                mpGlobalMap->min_id = pKF->frameID;
                mpGlobalMap->max_id = mpCurrentKF->frameID;
                //matcher.showMatch(mpCurrentKF,pKF,goodMatches);
            }
        }
        nCandidates++;
    }
    return success;
}

}
