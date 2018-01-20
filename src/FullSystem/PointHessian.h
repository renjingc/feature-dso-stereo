#pragma once
#define MAX_ACTIVE_FRAMES 100


#include "util/globalCalib.h"
#include "vector"

#include <iostream>
#include <fstream>
#include "util/NumType.h"
#include "util/settings.h"
#include "util/globalCalib.h"

#include "FullSystem/FrameHessian.h"
#include "FullSystem/FrameShell.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/ImmaturePoint.h"
#include "util/ImageAndExposure.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/eigen.hpp>



namespace fdso
{

struct FrameHessian;
struct PointHessian;

class ImmaturePoint;
class FrameShell;

class EFFrame;
class EFPoint;

class Feature;


// hessian component associated with one point.
/**
 * 每一个点的Hessian
 */
struct PointHessian
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  static int instanceCounter;

  //点的能量函数
  EFPoint* efPoint;

  // static values
  // 主导帧中这个点的模式灰度值
  float color[MAX_RES_PER_POINT];     // colors in host frame
  //主导帧中这个点的权重
  float weights[MAX_RES_PER_POINT];   // host-weights for respective residuals.

  //点的坐标
  float u, v;
  //点id
  int idx;
  //点的能量阈值
  float energyTH;
  //点对应的主导帧
  std::shared_ptr<FrameHessian> host;
  //是否有先验的深度值
  bool hasDepthPrior;

  //对应的特帧点
  std::shared_ptr<Feature> mF=nullptr;
  int feaMode=0;

  //类型
  float my_type;

  Vec3 mWorldPos;

  //逆深度
  float idepth_scaled;
  float idepth_zero_scaled;
  float idepth_zero;

  float idepth;
  float step;
  float step_backup;
  float idepth_backup;

  float nullspaces_scale;
  float idepth_hessian;
  float maxRelBaseline;
  int numGoodResiduals;

  enum PtStatus {ACTIVE = 0, INACTIVE, OUTLIER, OOB, MARGINALIZED};
  PtStatus status;

  inline void setPointStatus(PtStatus s) {status = s;}


  /**
   * [setIdepth description]
   * @param idepth [description]
   * 设置逆深度
   */
  inline void setIdepth(float idepth) {
    this->idepth = idepth;
    this->idepth_scaled = SCALE_IDEPTH * idepth;
  }

  /**
   * [setIdepthScaled description]
   * @param idepth_scaled [description]
   * 设置带尺度的逆深度
   */
  inline void setIdepthScaled(float idepth_scaled) {
    this->idepth = SCALE_IDEPTH_INVERSE * idepth_scaled;
    this->idepth_scaled = idepth_scaled;
  }

  /**
   * [setIdepthZero description]
   * @param idepth [description]
   */
  inline void setIdepthZero(float idepth) {
    idepth_zero = idepth;
    idepth_zero_scaled = SCALE_IDEPTH * idepth;
    nullspaces_scale = -(idepth * 1.001 - idepth / 1.001) * 500;
  }

  //只有好的残差（不是OOB不离群）。任意阶。
  std::vector<PointFrameResidual*> residuals;         // only contains good residuals (not OOB and not OUTLIER). Arbitrary order.
  //包含有关最后两个（的）残差的信息。帧。（[ 0 ] =最新的，[ 1 ] =前一个）。
  std::pair<PointFrameResidual*, ResState> lastResiduals[2];  // contains information about residuals to the last two (!) frames. ([0] = latest, [1] = the one before).


  void release();
  PointHessian(const std::shared_ptr<ImmaturePoint> rawPoint, CalibHessian* Hcalib);
  inline ~PointHessian() {assert(efPoint == 0); release(); instanceCounter--;}


  /**
   * [isOOB description]
   * @param  toKeep [description]
   * @param  toMarg [description]
   * @return        [description]
   * 是否是外点
   */
  inline bool isOOB(const std::vector<std::shared_ptr<FrameHessian>>& toKeep, const std::vector<std::shared_ptr<FrameHessian>>& toMarg) const
  {
    int visInToMarg = 0;
    //该点有目标帧被边缘化，visInToMarg个数++
    for (PointFrameResidual* r : residuals)
    {
      if (r->state_state != ResState::IN) continue;
      for (std::shared_ptr<FrameHessian> k : toMarg)
        if (r->target == k)
          visInToMarg++;
    }

    //残差数量够大３，１４，３，且无边缘化的残差数够小
    if ((int)residuals.size() >= setting_minGoodActiveResForMarg &&
        numGoodResiduals > setting_minGoodResForMarg + 10 &&
        (int)residuals.size() - visInToMarg < setting_minGoodActiveResForMarg)
      return true;

    if (lastResiduals[0].second == ResState::OOB) return true;
    if (residuals.size() < 2) return false;
    if (lastResiduals[0].second == ResState::OUTLIER && lastResiduals[1].second == ResState::OUTLIER) return true;
    return false;
  }


  /**
   * [isInlierNew description]
   * @return [description]
   * 是否是新的内点
   */
  inline bool isInlierNew()
  {
    return (int)residuals.size() >= setting_minGoodActiveResForMarg
           && numGoodResiduals >= setting_minGoodResForMarg;
  }

  void save(ofstream &fout);
  void load(ifstream &fin, vector<std::shared_ptr<FrameHessian>> &allKFs);

  void ComputeWorldPos();

};

}
