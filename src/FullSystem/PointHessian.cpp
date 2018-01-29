#include "PointHessian.h"

namespace fdso
{

/**
 * 点初始化状态为INACTIVE
 */
PointHessian::PointHessian(const ImmaturePoint* rawPoint, CalibHessian* Hcalib)
{
  //count++
  instanceCounter++;
  //设置点的主导帧
  host = rawPoint->host;
  //是否有先验深度
  hasDepthPrior = false;

  //逆深度的hessian矩阵
  idepth_hessian = 0;
  //最大相对baseline
  maxRelBaseline = 0;
  //好的残差
  numGoodResiduals = 0;

  // set static values & initialization.
  // 图像坐标
  u = rawPoint->u;
  v = rawPoint->v;
  assert(std::isfinite(rawPoint->idepth_max));
  //idepth_init = rawPoint->idepth_GT;

  //点的类型
  my_type = rawPoint->my_type;

  //设置逆深度
  setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min) * 0.5);
  //初始化点的状态为INACTIVE
  setPointStatus(PointHessian::INACTIVE);

  //SSE模式的点数
  int n = patternNum;
  memcpy(color, rawPoint->color, sizeof(float)*n);
  memcpy(weights, rawPoint->weights, sizeof(float)*n);

  //误差能量的阈值
  energyTH = rawPoint->energyTH;

  efPoint = 0;
}

/**
 * [PointHessian::release description]
 */
void PointHessian::release()
{
  if(mF)
    mF=nullptr;
  for (unsigned int i = 0; i < residuals.size(); i++) delete residuals[i];
  residuals.clear();
}

void PointHessian::ComputeWorldPos()
{
  if (!host)
    return;
  SE3 Twc = host->shell->camToWorldOpti;
  Vec3 Kip = 1.0 / this->idepth * Vec3(
               fxiG[0]  * this->u + cxiG[0],
             fyiG[0]  * this->v + cyiG[0],
             1);
  mWorldPos = Twc * Kip;
}

void PointHessian::save(ofstream &fout) {
  fout.write((char *) &this->idx, sizeof(this->idx));
  fout.write((char *) &this->status, sizeof(this->status));
}

void PointHessian::load(ifstream &fin, vector<FrameHessian*> &allKFs) 
{
  fin.read((char *) &this->idx, sizeof(this->idx));
  fin.read((char *) &this->status, sizeof(this->status));
}

}
