#ifndef OPTIMIZERPNP_H
#define OPTIMIZERPNP_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/eigen.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "Converter.h"

int PoseOptimization(const cv::Mat& mTcw, const std::vector<cv::KeyPoint>& mvKeys,
                                           const std::vector<cv::KeyPoint>& mvKeysRight,const std::vector<cv::KeyPoint>& mvKeysUn,
                                           const std::vector<cv::Point3f>& mvpMapPoints,
                                           const std::vector<float>& mvInvLevelSigma2,
                                           const float fx,const float fy,const float cx,const float cy,
                                           std::vector<bool>& mvbOutlier,
                                           cv::Mat& pose);

#endif 