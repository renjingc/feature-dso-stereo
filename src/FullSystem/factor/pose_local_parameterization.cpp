#include "pose_local_parameterization.h"

/*
 * 位姿误差
 */
bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
    //位置_p
    Eigen::Map<const Eigen::Vector3d> _p(x);
    //四元数_q
    Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

    //dp
    Eigen::Map<const Eigen::Vector3d> dp(delta);

    //相对旋转四元数
    Eigen::Quaterniond dq = Utility::deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

    //增量后的位置和四元数角度
    Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
    Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

    //位置相加
    p = _p + dp;
    //角度相乘
    q = (_q * dq).normalized();

    return true;
}
/*
 * 位姿计算雅克比
 */
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    //前6行为单位矩阵
    j.topRows<6>().setIdentity();
    //最后一行设为0,即尺度为0
    j.bottomRows<1>().setZero();

    return true;
}
