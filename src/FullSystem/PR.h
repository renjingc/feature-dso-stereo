#pragma once

#include "util/NumTypes.h"
#include "FullSystem/CalibHessian.h"

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/base_multi_edge.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/types/vertex_pointxyz.h>
#include <g2o/types/edge_pointxyz.h>

using namespace Eigen;
using namespace g2o;

using namespace fdso;

namespace fdso {

    struct Frame;

    // ---------------------------------------------------------------------------------------------------------
    // some g2o types will be used in pose graph
    class VertexPR : public BaseVertex<6, SE3> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        VertexPR() : BaseVertex<6, SE3>() {}

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        virtual void setToOriginImpl() override {
            _estimate = SE3();
        }

        virtual void oplusImpl(const double *update_) override {
            // P直接加，R右乘
            Vec6 update;
            update << update_[0], update_[1], update_[2], update_[3], update_[4], update_[5];
            _estimate = SE3::exp(update) * _estimate;

            /*
            _estimate.segment<3>(0) += Vec3(update_[0], update_[1], update_[2]);
            _estimate.segment<3>(3) = SO3::log(
                    SO3::exp(_estimate.segment<3>(3)) *
                    SO3::exp(Vec3(update_[3], update_[4], update_[5])));
                    */
        }

        inline Matrix3d R() const {
            return _estimate.so3().matrix();
        }

        inline Vector3d t() const {
            return _estimate.translation();
        }
    };

    /**
     * 逆深度地图点
     * _estimate 为逆深度
     */
    class VertexPointInvDepth : public BaseVertex<1, double> {

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        VertexPointInvDepth() : BaseVertex<1, double>() {};

        bool read(std::istream &is) { return true; }

        bool write(std::ostream &os) const { return true; }

        virtual void setToOriginImpl() {
            _estimate = 1.0;
        }

        virtual void oplusImpl(const double *update) {
            _estimate += update[0];
        }
    };


    // ---- Edges --------------------------------------------------------------------------------------------------

    /**
     * Edge of inverse depth prior for stereo-triangulated mappoints
     * Vertex: inverse depth map point
     *
     * Note: User should set the information matrix (inverse covariance) according to feature position uncertainty and baseline
     */
    class EdgeIDPPrior : public BaseUnaryEdge<1, double, VertexPointInvDepth> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeIDPPrior() : BaseUnaryEdge<1, double, VertexPointInvDepth>() {}

        virtual bool read(std::istream &is) override { return true; }

        virtual bool write(std::ostream &os) const override { return true; }

        virtual void computeError() override;

        // virtual void linearizeOplus() override;
    };

    /**
     * Odometry edge
     * err = T1.inv * T2
     */
    class EdgePR : public BaseBinaryEdge<6, SE3, VertexPR, VertexPR> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgePR() : BaseBinaryEdge<6, SE3, VertexPR, VertexPR>() {}

        virtual bool read(std::istream &is) override { return true; }

        virtual bool write(std::ostream &os) const override { return true; }

        virtual void computeError() override {
            SE3 v1 = (static_cast<VertexPR *> (_vertices[0]))->estimate();
            SE3 v2 = (static_cast<VertexPR *> (_vertices[1]))->estimate();
            _error = (_measurement.inverse() * v1 * v2.inverse()).log();
        };

        /*
        virtual void linearizeOplus() override {
            SE3 v1 = (static_cast<VertexPR *> (_vertices[0]))->estimate();
            SE3 v2 = (static_cast<VertexPR *> (_vertices[1]))->estimate();
            Mat66 J = JRInv(SE3::exp(_error));
            // 尝试把J近似为I？
            _jacobianOplusXi = -J * v2.inverse().Adj();
            _jacobianOplusXj = J * v2.inverse().Adj();
        }
         */

        inline Mat66 JRInv(const SE3 &e) {
            Mat66 J;
            J.block(0, 0, 3, 3) = SO3::hat(e.so3().log());
            J.block(0, 3, 3, 3) = SO3::hat(e.translation());
            J.block(3, 0, 3, 3) = Eigen::Matrix3d::Zero(3, 3);
            J.block(3, 3, 3, 3) = SO3::hat(e.so3().log());
            J = J * 0.5 + Mat66::Identity();
            return J;
        }
    };

    /**
    * Edge of reprojection error in one frame. Contain 3 vectices
    * Vertex 0: inverse depth map point
    * Veretx 1: Host KF PR
    * Vertex 2: Target KF PR
    **/
    class EdgePRIDP : public BaseMultiEdge<2, Vec2> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        // give the normalized x, y and camera intrinsics
        EdgePRIDP(double x, double y, CalibHessian* calib) : BaseMultiEdge<2, Vec2>() {
            resize(3);
            this->x = x;
            this->y = y;
            cam = calib;
        }

        bool read(std::istream &is) override { return true; }

        bool write(std::ostream &os) const override { return true; }

        virtual void computeError() override;

        // virtual void linearizeOplus() override;

        bool isDepthValid() {
            return dynamic_cast<const VertexPointInvDepth *>( _vertices[0])->estimate() > 0;
        }

    protected:

        // [x,y] in normalized image plane in reference KF
        double x = 0, y = 0;
        bool linearized = false;

        CalibHessian* cam;
    };

    /*
    G2O_REGISTER_ACTION(VertexPR);
    G2O_REGISTER_ACTION(VertexPointInvDepth);
    G2O_REGISTER_ACTION(EdgeIDPPrior);
    G2O_REGISTER_ACTION(EdgePRIDP);
     */
}

