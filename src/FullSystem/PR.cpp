#include "PR.h"
#include "FullSystem/GlobalCalib.h"
#include "FullSystem/ResidualProjections.h"

using namespace fdso;

namespace fdso
{
    void EdgeIDPPrior::computeError()
    {
        const VertexPointInvDepth *vIDP = static_cast<const VertexPointInvDepth *>(_vertices[0]);
        _error(0) = vIDP->estimate() - _measurement;
    }

    /*
    void EdgeIDPPrior::linearizeOplus() {
        _jacobianOplusXi.setZero();
        _jacobianOplusXi(0) = 1;
    }
     */

    /**
     * Erorr = pi(Px)-obs
     */
    void EdgePRIDP::computeError()
    {
        const VertexPointInvDepth *vIDP = dynamic_cast<const VertexPointInvDepth *>( _vertices[0]);
        const VertexPR *vPR0 = dynamic_cast<const VertexPR *>(_vertices[1]);
        const VertexPR *vPRi = dynamic_cast<const VertexPR *>(_vertices[2]);

        // point inverse depth in reference KF
        double rho = vIDP->estimate();
        if (rho < 1e-6) {
            // LOG(WARNING) << "Inv depth should not be negative: " << rho << endl;
            return;
        }

        // point coordinate in reference KF, body
        Vec3 P0(x, y, 1.0);
        P0 = P0 * (1.0f / rho);
        Vec3 Pw = vPR0->estimate().inverse() * P0;
        Vec3 Pi = vPRi->estimate() * Pw;

        if (Pi[2] < 0) {
            // LOG(WARNING) << "projected depth should not be negative: " << Pi.transpose() << endl;
            return;
        }

        double xi = Pi[0] / Pi[2];
        double yi = Pi[1] / Pi[2];
        double u = cam->fxl() * xi + cam->cxl();
        double v = cam->fyl() * yi + cam->cyl();
        _error = Vec2(u, v) - _measurement;
    }

    /*
    void EdgePRIDP::linearizeOplus() {

        // if ( linearized == false ) {

            const VertexPointInvDepth *vIDP = dynamic_cast<const VertexPointInvDepth *>(_vertices[0]);
            const VertexPR *vPR0 = dynamic_cast<const VertexPR *>(_vertices[1]);
            const VertexPR *vPRi = dynamic_cast<const VertexPR *>(_vertices[2]);

            // 李代数到xy的导数
            Vec6f d_xi_x, d_xi_y;
            Matrix<double,2,6> J_xi;

            // xy 到 idepth 的导数
            float d_d_x, d_d_y;

            // NOTE u = X/Z, v=Y/Z in target
            Vec3 KliP(x,y,1);

            SE3 Tll = vPRi->estimate() * vPR0->estimate().inverse();
            Mat33 R = Tll.rotationMatrix();
            Vec3 t = Tll.translation();

            double idepth = vIDP->estimate();

            Vec3 ptp = R * KliP + t * idepth;
            double drescale = 1.0f / ptp[2];
            double new_idepth = idepth * drescale;

            if (ptp[2] < 0)
                return;

            double u = ptp[0] * drescale;
            double v = ptp[1] * drescale;

            // 各种导数
            // diff d_idepth
            d_d_x = drescale * (t[0] - t[2] * u) * cam->fxl();
            d_d_y = drescale * (t[1] - t[2] * v) * cam->fyl();

            // xy到李代数的导数，形式见十四讲
            J_xi(0,0) = new_idepth * cam->fxl();
            J_xi(0,1) = 0;
            J_xi(0,2) = -new_idepth * u * cam->fxl();
            J_xi(0,3) = -u * v * cam->fxl();
            J_xi(0,4) = (1 + u * u) * cam->fxl();
            J_xi(0,5) = -v * cam->fxl();

            J_xi(1,0) = 0;
            J_xi(1,1) = new_idepth * cam->fyl();
            J_xi(1,2) = -new_idepth * v * cam->fyl();
            J_xi(1,3) = -(1 + v * v) * cam->fyl();
            J_xi(1,4) = u * v * cam->fyl();
            J_xi(1,5) = u * cam->fyl();

            _jacobianOplus[0] = Vec2( d_d_x, d_d_y );
            _jacobianOplus[1] = -J_xi;
            _jacobianOplus[2] = J_xi*Tll.Adj();

            linearized=true;
        // }


        /*
        // 1. J_e_rho, 2x1
        // Vector3d J_pi_rho = Rcic0 * (-d * P0);
        Vector3d J_pi_rho = Rcic0 * (-d * P0); // (xiang) this should be squared?
        _jacobianOplus[0] = -Jpi * J_pi_rho;

        // 2. J_e_pr0, 2x6
        Matrix3d J_pi_t0 = RiT;
        Matrix3d J_pi_r0 = -Rcic0 * SO3::hat(P0);
        Matrix<double, 3, 6> J_pi_pr0;
        J_pi_pr0.topLeftCorner(3, 3) = J_pi_t0;
        J_pi_pr0.topRightCorner(3, 3) = J_pi_r0;
        _jacobianOplus[1] = -Jpi * J_pi_pr0;

        // 3. J_e_pri, 2x6
        Matrix3d J_pi_ti = -RiT;
        Vector3d taux = RiT * (R0 * P0 + t0 - ti);
        Matrix3d J_pi_ri = SO3::hat(taux);
        Matrix<double, 3, 6> J_pi_pri;
        J_pi_pri.topLeftCorner(3, 3) = J_pi_ti;
        J_pi_pri.topRightCorner(3, 3) = J_pi_ri;
        _jacobianOplus[2] = -Jpi * J_pi_pri;
    }
         */

}
