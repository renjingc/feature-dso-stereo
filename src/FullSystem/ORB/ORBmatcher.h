#ifndef ORBMATCHER_H
#define ORBMATCHER_H

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include <map>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include "ORBextractor.h"

#include "util/FrameShell.h"
#include "util/NumType.h"

// 匹配相关的算法，包括特征点的匹配法和直接法的匹配法
class Matcher
{
public:

  // 特征匹配的参数，从orb-slam2中拷贝
  struct Options {
    int th_high = 100;       // 这两个在搜索匹配时用
    int th_low = 50;         // 低阈值
    float knnRatio = 0.9;    // knn 时的比例，在SearchByBoW中用于计算最优和次优的差别

    bool checkOrientation = false;  // 除了检测描述之外是否检查旋转

    float initMatchRatio = 3.0;     // 初始化时的比例
    int init_low = 30;              // 这两个在初始化时用于检测光流结果是否正确
    int init_high = 80;

    double _max_alignment_motion = 0.2; // 稀疏匹配中能够接受的最大的运动
    double _epipolar_dsqr = 1e-4;   // 寻找三角化点时，极线检查时的最大误差阈值

  } _options;


  static const int HISTO_LENGTH = 30;  // 旋转直方图的size

  Matcher();
  ~Matcher();

  void SetTCR( const SE3& TCR )
  {
    _TCR_esti = TCR;
  }

  // 特征点法的匹配
  // Computes the Hamming distance between two ORB descriptors
  // 两个描述子之间的Hamming距离，它们必须是1x32的ORB描述
  static int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

  // 搜索特征点以建立三角化结果
  // 给定两帧之间的 Essential，计算 matched points
  // 结果中的 kf1 关键点状态必须没有三角化，kf2中则无所谓
  int SearchForTriangulation(
    FrameShell* kf1, FrameShell* kf2, const Eigen::Matrix3d& E12,
    std::vector< std::pair<int, int> >& matched_points,
    const bool& onlyStereo = false
  );

  // 在Keyframe之间搜索匹配情况，利用BoW加速
  int SearchByBoW( FrameShell * kf1, FrameShell* kf2, std::map<int, int>& matches );

  // 在KeyFrame和地图之间搜索匹配情况
  // int SearchByProjection(Frame *F, const std::vector<MapPoint*> &vpMapPoints, const float th=3);

  // 计算两个帧之间的特征描述是否一致
  // 这是在初始化里用的。初始化使用了光流跟踪了一些点，但我们没法保证跟踪成功，所以需要再检查一遍它们的描述量
  // 第三个参数内指定了光流追踪的match，如果描述不符合，就会从里面剔除
  int CheckFrameDescriptors( FrameShell* frame1, FrameShell* frame2, std::list<std::pair<int, int>>& matches );


  // ****************************************************************************************************
  // 直接法的匹配

  // 用直接法判断能否从在当前图像上找到某地图点的投影
  //bool FindDirectProjection( FrameShell* ref, FrameShell* curr, MapPoint* mp, Vector2d& px_curr, int& search_level );
  // 重载: 已知feature的情况（尚未建立地图点时）
  //bool FindDirectProjection( FrameShell* ref, FrameShell* curr, Feature* fea_ref, Vector2d& px_curr, int& search_level );

  // model based sparse image alignment
  // 通过参照帧中观测到的3D点，预测当前帧的pose，稀疏直接法
  bool SparseImageAlignment( FrameShell* ref, FrameShell* current );

  SE3 GetTCR() const
  {
    return _TCR_esti;
  }

private:
  // 内部函数
  // 计算旋转直方图中三个最大值
  void ComputeThreeMaxima( std::vector<int>* histo, const int L, int& ind1, int& ind2, int& ind3 );

  bool CheckDistEpipolarLine( const Eigen::Vector3d& pt1, const Eigen::Vector3d& pt2, const Eigen::Matrix3d& E12 );

  // 对每层金字塔计算的 image alignment
  bool SparseImageAlignmentInPyramid( FrameShell* ref, FrameShell* current, int pyramid );


  void GetWarpAffineMatrix (
    const FrameShell* ref,
    const FrameShell* curr,
    const Eigen::Vector2d& px_ref,
    const Eigen::Vector3d& pt_ref,
    const int & level,
    const SE3& TCR,
    Eigen::Matrix2d& ACR
  );

  // perform affine warp
  void WarpAffine (
    const Eigen::Matrix2d& ACR,
    const cv::Mat& img_ref,
    const Eigen::Vector2d& px_ref,
    const int& level_ref,
    const int& search_level,
    const int& half_patch_size,
    uint8_t* patch
  );

  // 计算最好的金字塔层数
  // 选择一个分辨率，使得warp不要太大
  inline int GetBestSearchLevel (
    const Eigen::Matrix2d& ACR,
    const int& max_level )
  {
    int search_level = 0;
    double D = ACR.determinant();
    while ( D > 3.0 && search_level < max_level ) {
      search_level += 1;
      D *= 0.25;
    }
    return search_level;
  }

  // 计算参照帧中的图像块
  void PrecomputeReferencePatches( FrameShell* ref, int level );

private:
  // Data

  // 匹配局部地图用的 patch, 默认8x8
  uchar _patch[WarpPatchSize * WarpPatchSize];
  // 带边界的，左右各1个像素
  uchar _patch_with_border[(WarpPatchSize + 2) * (WarpPatchSize + 2)];

  std::vector<uchar*> _patches_align;      // 等待推倒的patches
  std::vector<Vector2d*> _duv_ref;         // Reference 像素梯度
  SparseImgAlign* _align;
  SE3 _TCR_esti;      // 待估计的TCR

};

}


#endif // ORBMATCHER_H