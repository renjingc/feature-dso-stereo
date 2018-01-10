/**
* This file is part of the implementation of our papers: 
* [1] Yonggen Ling, Manohar Kuse, and Shaojie Shen, "Direct Edge Alignment-Based Visual-Inertial Fusion for Tracking of Aggressive Motions", Autonomous Robots, 2017.
* [2] Yonggen Ling and Shaojie Shen, "Aggresive Quadrotor Flight Using Dense Visual-Inertial Fusion", in Proc. of the IEEE Intl. Conf. on Robot. and Autom., 2016.
* [3] Yonggen Ling and Shaojie Shen, "Dense Visual-Inertial Odometry for Tracking of Aggressive Motions", in Proc. of the IEEE International Conference on Robotics and Biomimetics 2015.
*
*
* For more information see <https://github.com/ygling2008/direct_edge_imu>
*
* This code is a free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This code is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this code. If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/local_parameterization.h>
#include "utility.h"

#include "sophus/sim3.hpp"
#include "sophus/se3.hpp"

class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
};

///// PoseBlockSize can only be
///// 7 ( translation vector + quaternion) or
///// 6 (translation vector + rotation vector)
///// 6 (se3)
//template<int PoseBlockSize>
//class PoseSE3Parameterization : public ceres::LocalParameterization {
//public:
//    PoseSE3Parameterization() {}
//    virtual ~PoseSE3Parameterization() {}
//    virtual bool Plus(const double* x,
//                      const double* delta,
//                      double* x_plus_delta) const;
//    virtual bool ComputeJacobian(const double* x,
//                                 double* jacobian) const;
//    virtual int GlobalSize() const { return PoseBlockSize; }
//    virtual int LocalSize() const { return 6; }
//};

class LocalParameterizationSE3 : public ceres::LocalParameterization {
 public:
  virtual ~LocalParameterizationSE3() {}

  // SE3 plus operation for Ceres
  //
  //  T * exp(x)
  //
  virtual bool Plus(double const* T_raw, double const* delta_raw,
                    double* T_plus_delta_raw) const {
    Eigen::Map<Sophus::SE3 const> const T(T_raw);
    Eigen::Map<Eigen::Matrix<double,6,1> const> const delta(delta_raw);
    Eigen::Map<Sophus::SE3> T_plus_delta(T_plus_delta_raw);
    T_plus_delta = Sophus::SE3::exp(delta) * T;
    return true;
  }

  // Jacobian of SE3 plus operation for Ceres
  //
  // dx T * exp(x)  with  x=0
  //
  virtual bool ComputeJacobian(double const* T_raw,
                               double* jacobian_raw) const {
    Eigen::Map<Sophus::SE3 const> T(T_raw);
    Eigen::Map<Eigen::Matrix<double, 6, 7> > jacobian(jacobian_raw);
    jacobian.setIdentity();

    return true;
  }

  virtual int GlobalSize() const { return Sophus::SE3::DoF; }

  virtual int LocalSize() const { return Sophus::SE3::DoF; }
};
