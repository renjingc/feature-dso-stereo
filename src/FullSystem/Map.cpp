#include "Map.h"

#include "FullSystem/FrameHessian.h"
#include "FullSystem/PointHessian.h"
#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/CalibHessian.h"
#include "FullSystem/PR.h"

// need g2o stuffs here and only here
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/linear_solver_eigen.h>
#include <g2o/core/robust_kernel_impl.h>

using namespace std;
using namespace fdso;

namespace fdso {

//增加关键帧
void Map::addKeyFrame(std::shared_ptr<FrameHessian> kf)
{
    std::unique_lock<std::mutex> mapLock(mapMutex);
    if (frames.find(kf) == frames.end())
    {
        frames.insert(kf);
    }
}

//全部优化
void Map::optimizeALLKFs()
{
    {
        //锁上位姿图优化
        std::unique_lock<std::mutex> lock(mutexPoseGraph);

        //是否已经在运行
        if (poseGraphRunning)
            return; // is already running ...
        // if not, starts it
        //开始运行
        poseGraphRunning = true;
        // lock frames to prevent adding new kfs
        //锁上地图
        std::unique_lock<std::mutex> mapLock(mapMutex);
        //当前最新的关键帧
        framesOpti = frames;
        currentKF = *frames.rbegin();
    }
    //  start the pose graph thread
    //开始优化
    std::thread th = thread(&Map::runPoseGraphOptimization, this);
    th.detach();    // it will set posegraphrunning to false when returns
}

void Map::runPoseGraphOptimization()
{
    //位姿优化
    LOG(INFO) << "start pose graph thread!" << endl;
    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolverX::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>();
    g2o::BlockSolverX *solver_ptr = new g2o::BlockSolverX(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    // 关键帧顶点
    int maxKFid = 0;
    int cntEdgePR = 0;

    for (std::shared_ptr<FrameHessian> fh : framesOpti)
    {
        // 每个KF只有P+R
        int idKF = fh->frameID;
        if (idKF > maxKFid)
        {
            maxKFid = idKF;
        }

        // P+R
        VertexPR *vPR = new VertexPR();
        vPR->setEstimate(fh->shell->camToWorldOpti.inverse());
        vPR->setId(idKF);
        optimizer.addVertex(vPR);

        // fix the last one since we don't want to affect the frames in window
        if (fh == currentKF)
        {
            vPR->setFixed(true);
        }
    }

    // edges
    for (std::shared_ptr<FrameHessian> fh : framesOpti)
    {
        unique_lock<mutex> lock(fh->mMutexPoseRel);
        for (auto &rel : fh->mPoseRel)
        {
            VertexPR *vPR1 = (VertexPR *) optimizer.vertex(fh->frameID);
            VertexPR *vPR2 = (VertexPR *) optimizer.vertex(rel.first->frameID);

            EdgePR *edgePR = new EdgePR();
            if (vPR1 == nullptr || vPR2 == nullptr)
                continue;
            edgePR->setVertex(0, vPR1);
            edgePR->setVertex(1, vPR2);
            edgePR->setMeasurement(rel.second);
            edgePR->setInformation(Mat66::Identity());
            optimizer.addEdge(edgePR);
            cntEdgePR++;
        }
    }

    optimizer.initializeOptimization();
    optimizer.optimize(20);

    // recover the pose and points estimation
    for (std::shared_ptr<FrameHessian> frame : framesOpti)
    {
        VertexPR *vPR = (VertexPR *) optimizer.vertex(frame->frameID);
        SE3 Tcw = vPR->estimate();
        frame->shell->camToWorldOpti = Tcw.inverse();

        // reset the map point world position because we've changed the keyframe pose
        for (auto &point : frame->pointHessians)
        {
            point->ComputeWorldPos();
        }
        for (auto &point : frame->pointHessiansOut)
        {
            point->ComputeWorldPos();
        }
        for (auto &point : frame->pointHessiansMarginalized)
        {
            point->ComputeWorldPos();
        }
    }

    //优化结束
    poseGraphRunning = false;
}
}
