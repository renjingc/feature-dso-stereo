#pragma once

#include "util/NumType.h"
#include "util/globalCalib.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/eigen.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

namespace fdso
{

bool checkEstimatedPose(int num_inliers_,SE3 T_c_r_estimated_);

int PoseOptimization(const cv::Mat& mTcw, const std::vector<cv::KeyPoint>& mvKeys,
                                           const std::vector<cv::KeyPoint>& mvKeysRight,const std::vector<cv::KeyPoint>& mvKeysUn,
                                           const std::vector<cv::Point3f>& mvpMapPoints,
                                           const std::vector<float>& mvInvLevelSigma2,
                                           const float fx,const float fy,const float cx,const float cy,
                                           std::vector<bool>& mvbOutlier,
                                           cv::Mat& pose);

bool pnpCv(const SE3 initT, const std::vector<cv::Point2f>& p2d,
          const std::vector<cv::Point3f>& p3d,
          const cv::Mat K,
          int& cntInliers,
          std::vector<bool>& mvbOutlier,
          SE3& outT);
}