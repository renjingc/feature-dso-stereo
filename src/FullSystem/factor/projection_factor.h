#ifndef PROJECTION_FACTOR_H
#define PROJECTION_FACTOR_H

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"

/*
 * 重投影的误差因子，这里使用SizedCostFunstion
 * 残差为2维，优化变量为7,7,7,1维，共4个，i帧位姿，j帧位姿，相机相对与imu的位姿，深度
 */
class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1>
{
  public:
    ProjectionFactor(const Eigen::Vector3d &_pts_i, const Eigen::Vector3d &_pts_j);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);

    //i和j的特征点
    Eigen::Vector3d pts_i, pts_j;
    //切线，2*3
    Eigen::Matrix<double, 2, 3> tangent_base;
    //协方差矩阵
    static Eigen::Matrix2d sqrt_info;
    //总时间
    static double sum_t;
};

#endif
