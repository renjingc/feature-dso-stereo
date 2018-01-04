#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

/*
 * ceres的位姿误差，这里使用LocalParameterization
 * 需要定义Plus，ComputeJacobian，GlobalSize，LocalSize
 * 超参数，选择参数化来消除误差的零方向，即某一方向的移动是无用的，则只在某些方向更新，则用这个
 * Plus:⊞(x,deltaX)
 * ComputeJacobian: J=∂/∂Δx ⊞(x,Δx)∣∣∣Δx=0
 */
class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
};
