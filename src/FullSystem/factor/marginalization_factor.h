#pragma once

#include <ros/ros.h>
#include <ros/console.h>
#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

//四个线程
const int NUM_THREADS = 4;

/**
 * @brief The ResidualBlockInfo struct
 * 残差块信息
 */
struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;

    //原始的雅克比
    double **raw_jacobians;
    //雅克比
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    //残差
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

/**
 * @brief The ThreadsStruct struct
 */
struct ThreadsStruct
{
    //因子
    std::vector<ResidualBlockInfo *> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    //参数块大小
    std::unordered_map<long, int> parameter_block_size; //global size
    //参数块id
    std::unordered_map<long, int> parameter_block_idx; //local size
};

/**
 * @brief The MarginalizationInfo class
 * 边缘化信息
 */
class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    int localSize(int size) const;
    int globalSize(int size) const;
    //增加残差块信息
    void addResidualBlockInfo(ResidualBlockInfo *residual_block_info);
    //之前的边缘化
    void preMarginalize();
    //边缘化
    void marginalize();
    //获取参数块
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    //残差块因子
    std::vector<ResidualBlockInfo *> factors;
    //n为残差大小维度
    int m, n;
    //每个参数块大小，前面是地址，后面是大小
    std::unordered_map<long, int> parameter_block_size; //global size
    //参数全部大小
    int sum_block_size;
    //每个参数块id
    std::unordered_map<long, int> parameter_block_idx; //local size
    //参数块数据，首地址，数据
    std::unordered_map<long, double *> parameter_block_data;

    std::vector<int> keep_block_size; //global size
    std::vector<int> keep_block_idx;  //local size
    std::vector<double *> keep_block_data;

    Eigen::MatrixXd linearized_jacobians;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;

};

/**
 * @brief The MarginalizationFactor class
 * 边缘化因子
 * 输入边缘化信息
 */
class MarginalizationFactor : public ceres::CostFunction
{
  public:
    MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    MarginalizationInfo* marginalization_info;
};
