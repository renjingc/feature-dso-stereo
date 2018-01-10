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

#include "pose_local_parameterization.h"

bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    Eigen::Map<const Eigen::Vector3d> _p(x);
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    Eigen::Map<const Eigen::Vector3d> dp(delta);

    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    p = _p + dp;
    q = (_q * dq).normalized();

    return true;
}
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}


//template<>
//bool PoseSE3Parameterization<7>::Plus(const double *x, const double *delta, double *x_plus_delta) const
//{
//    Eigen::Map<const Eigen::Vector3d> trans(x);
//    Sophus::SE3 se3_delta = Sophus::SE3::exp(Eigen::Map<const Eigen::Vector6d>(delta));

//    Eigen::Map<const Eigen::Quaterniond> quaterd(x+3);
//    Eigen::Map<Eigen::Quaterniond> quaterd_plus(x_plus_delta+3);
//    Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta);

//    quaterd_plus = se3_delta.rotation() * quaterd;
//    trans_plus = se3_delta.rotation() * trans + se3_delta.translation();

//    return true;
//}

//template<>
//bool PoseSE3Parameterization<7>::ComputeJacobian(const double *x, double *jacobian) const
//{
//    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > J(jacobian);
//    J.setZero();
//    J.block<6,6>(0, 0).setIdentity();
//    return true;
//}



//template<>
//bool PoseSE3Parameterization<6>::Plus(const double *x, const double *delta, double *x_plus_delta) const
//{
//    Eigen::Map<const Eigen::Vector3d> trans(x);
//    Sophus::SE3 se3_delta = SE3::exp(Eigen::Map<const Eigen::Vector6d>(delta));

//    Eigen::Quaterniond quaterd_plus = se3_delta.rotation() * toQuaterniond(Eigen::Map<const Eigen::Vector3d>(x+3));
//    Eigen::Map<Eigen::Vector3d> angles_plus(x_plus_delta+3);
//    angles_plus = toAngleAxis(quaterd_plus);

//    Eigen::Map<Eigen::Vector3d> trans_plus(x_plus_delta);
//    trans_plus = se3_delta.rotation() * trans + se3_delta.translation();
//    return true;
//}

//template<>
//bool PoseSE3Parameterization<6>::ComputeJacobian(const double *x, double *jacobian) const
//{
//    Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor> > J(jacobian);
//    J.setIdentity();
//    return true;
//}


