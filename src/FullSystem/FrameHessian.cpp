#include "FullSystem/FrameHessian.h"
#include "FullSystem/FrameShell.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

namespace fdso
{


/**
 * [FrameHessian::setStateZero description]
 * @param state_zero [description]
 * 设置帧的初始状态
 */
void FrameHessian::setStateZero(Vec10 state_zero)
{
  assert(state_zero.head<6>().squaredNorm() < 1e-20);

  this->state_zero = state_zero;


  for (int i = 0; i < 6; i++)
  {
    Vec6 eps; eps.setZero(); eps[i] = 1e-3;
    SE3 EepsP = Sophus::SE3::exp(eps);
    SE3 EepsM = Sophus::SE3::exp(-eps);
    SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
    SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
    nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);
  }
  //nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
  //nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

  // scale change
  // 小小的改变
  SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
  w2c_leftEps_P_x0.translation() *= 1.00001;
  w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
  SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
  w2c_leftEps_M_x0.translation() /= 1.00001;
  w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
  nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log()) / (2e-3);


  nullspaces_affine.setZero();
  nullspaces_affine.topLeftCorner<2, 1>()  = Vec2(1, 0);
  assert(ab_exposure > 0);
  nullspaces_affine.topRightCorner<2, 1>() = Vec2(0, expf(aff_g2l_0().a) * ab_exposure);
};

/**
 * [FrameHessian::release description]
 */
void FrameHessian::release()
{
  // DELETE POINT
  // DELETE RESIDUAL
  for (unsigned int i = 0; i < pointHessians.size(); i++) delete pointHessians[i];
  for (unsigned int i = 0; i < pointHessiansMarginalized.size(); i++) delete pointHessiansMarginalized[i];
  for (unsigned int i = 0; i < pointHessiansOut.size(); i++) delete pointHessiansOut[i];
  for (unsigned int i = 0; i < immaturePoints.size(); i++) delete immaturePoints[i];

  pointHessians.clear();
  pointHessiansMarginalized.clear();
  pointHessiansOut.clear();
  immaturePoints.clear();
}

void FrameHessian::ComputeBoW(ORBVocabulary* _vocab)
{
  if ( _vocab != nullptr && _bow_vec.empty() )
  {
//        _bow_vec =new DBoW3::BowVector();
//        _feature_vec = new DBoW3::FeatureVector();
    std::vector<cv::Mat> alldesp;
    for ( Feature* fea : _features )
    {
      alldesp.push_back(fea->_desc);
    }
    _vocab->transform( alldesp, _bow_vec, _feature_vec, 4);
  }
}

/**
 * [FrameHessian::makeImages description]
 * @param color  [description]
 * @param HCalib [description]
 * 创建该帧的每个点的xy梯度值
 */
void FrameHessian::makeImages(ImageAndExposure* imageE, CalibHessian* HCalib)
{
  float* color = imageE->image;
  image = imageE->toMat();
  _pyramid.resize( pyrLevelsUsed );
  _pyramid[0] = image;
  for ( size_t i = 1; i < pyrLevelsUsed; i++ )
  {
    // 在CV里使用down构造分辨率更低的图像，中间有高斯模糊化以降低噪声
    // 请记得第0层是原始分辨率
    cv::pyrDown( _pyramid[i - 1], _pyramid[i] );
  }
  //遍历每一层金字塔
  for (int i = 0; i < pyrLevelsUsed; i++)
  {
    //当前帧每层需要的图像,[0]为原始图像的灰度值，[1]为x方向的梯度值，[2]为y方向的梯度值
    dIp[i] = new Eigen::Vector3f[wG[i]*hG[i]];
    //当前帧的每个点的xy方向的梯度平方和
    absSquaredGrad[i] = new float[wG[i]*hG[i]];
  }
  //原始大小的图像
  dI = dIp[0];

  // make d0
  int w = wG[0];
  int h = hG[0];
  //di为每一个点的像素值
  for (int i = 0; i < w * h; i++)
    dI[i][0] = color[i];

  //六层的金字塔
  for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
  {
    //获取每层金字塔的大小
    int wl = wG[lvl], hl = hG[lvl];
    //当前层用于跟踪和初始化的图
    Eigen::Vector3f* dI_l = dIp[lvl];
    //当前层的绝对正方形梯度值
    float* dabs_l = absSquaredGrad[lvl];

    if (lvl > 0)
    {
      //上一层的层号
      int lvlm1 = lvl - 1;
      //上一层的宽
      int wlm1 = wG[lvlm1];
      //上一层的用于计算的图
      Eigen::Vector3f* dI_lm = dIp[lvlm1];

      for (int y = 0; y < hl; y++)
        for (int x = 0; x < wl; x++)
        {
          //四周像素的平均值,即越高层则图像越小，则每个点的值由其四周四个点平均得到
          dI_l[x + y * wl][0] = 0.25f * (dI_lm[2 * x   + 2 * y * wlm1][0] +
                                         dI_lm[2 * x + 1 + 2 * y * wlm1][0] +
                                         dI_lm[2 * x   + 2 * y * wlm1 + wlm1][0] +
                                         dI_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);
        }
    }
    //当前层每个点的梯度
    for (int idx = wl; idx < wl * (hl - 1); idx++)
    {
      //每个点的x和y方向的梯度
      float dx = 0.5f * (dI_l[idx + 1][0] - dI_l[idx - 1][0]);
      float dy = 0.5f * (dI_l[idx + wl][0] - dI_l[idx - wl][0]);

      //得到每个点xy方向梯度值
      if (!std::isfinite(dx)) dx = 0;
      if (!std::isfinite(dy)) dy = 0;

      //当前两个方向的梯度和
      dI_l[idx][1] = dx;
      dI_l[idx][2] = dy;

      dabs_l[idx] = dx * dx + dy * dy;

      //每个点有gamma矫正权重，且矫正参数不为0
      if (setting_gammaWeightsPixelSelect == 1 && HCalib != 0)
      {
        //获取每个点的gamma矫正后的灰度值
        float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
        //得到最终的每个点的梯度值
        //每个点的梯度平方和*矫正后的灰度值
        dabs_l[idx] *= gw * gw; // convert to gradient of original color space (before removing response).
      }
    }
  }
}

/**
 * [FrameFramePrecalc::set description]
 * @param host   [description]
 * @param target [description]
 * @param HCalib [description]
 * 设置主导帧，目标帧，内参
 */
void FrameFramePrecalc::set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib )
{
  // printf("whether this->host is NULL: yes is 1, no is 0. Answer: %x\n", this);
  //设置主导帧
  this->host = host;
  //设置目标真帧
  this->target = target;

  //两帧位姿变换
  SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
  //旋转
  PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
  //平移
  PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();

  //两帧位姿变换
  SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
  PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
  PRE_tTll = (leftToLeft.translation()).cast<float>();
  distanceLL = leftToLeft.translation().norm();

  //设置内参
  Mat33f K = Mat33f::Zero();
  K(0, 0) = HCalib->fxl();
  K(1, 1) = HCalib->fyl();
  K(0, 2) = HCalib->cxl();
  K(1, 2) = HCalib->cyl();
  K(2, 2) = 1;

  //K*R*K‘
  PRE_KRKiTll = K * PRE_RTll * K.inverse();
  //R*K'
  PRE_RKiTll = PRE_RTll * K.inverse();
  //K*t
  PRE_KtTll = K * PRE_tTll;

  //根据两帧的曝光时间和a和b，计算两帧之间的a和b,光度线性变换
  PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<float>();
  PRE_b0_mode = host->aff_g2l_0().b;
}


/**
 * { item_description }
 * 获取当前帧的连接
 */
set<FrameHessian*> FrameHessian::GetConnectedKeyFrames()
{
  set<FrameHessian*> connectedFrames;
  for (auto &rel : mPoseRel)
    connectedFrames.insert(rel.first);
  return connectedFrames;
}

}

