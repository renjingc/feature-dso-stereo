/**
* This file is part of DSO.
*
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "IOWrapper/ImageRW.h"
#include <algorithm>

namespace fdso
{

/**
 *
 */
CoarseTracker::CoarseTracker(int ww, int hh) : lastRef_aff_g2l(0, 0)
{
    // make coarse tracking templates.
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
    {
        //设置每层的图像大小
        int wl = ww >> lvl;
        int hl = hh >> lvl;
        //设置每层的逆深度大小
        idepth[lvl] = new float[wl * hl];
        //设置每层的权重
        weightSums[lvl] = new float[wl * hl];
        weightSums_bak[lvl] = new float[wl * hl];

        pc_u[lvl] = new float[wl * hl];
        pc_v[lvl] = new float[wl * hl];
        pc_idepth[lvl] = new float[wl * hl];
        pc_color[lvl] = new float[wl * hl];
    }

    // warped buffers
    buf_warped_idepth = new float[ww * hh];
    buf_warped_u = new float[ww * hh];
    buf_warped_v = new float[ww * hh];
    buf_warped_dx = new float[ww * hh];
    buf_warped_dy = new float[ww * hh];
    buf_warped_residual = new float[ww * hh];
    buf_warped_weight = new float[ww * hh];
    buf_warped_refColor = new float[ww * hh];


    //新的一帧
    newFrame = 0;
    //参考帧
    lastRef = 0;
    //是否输出中间结果
    debugPlot = debugPrint = true;
    //
    w[0] = h[0] = 0;
    //参考帧ID
    refFrameID = -1;
}

/**
 *
 */
CoarseTracker::~CoarseTracker()
{
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
    {
        delete[] idepth[lvl];
        delete[] weightSums[lvl];
        delete[] weightSums_bak[lvl];

        delete[] pc_u[lvl];
        delete[] pc_v[lvl] ;
        delete[] pc_idepth[lvl];
        delete[] pc_color[lvl];
    }

    delete[]  buf_warped_idepth;
    delete[]  buf_warped_u;
    delete[]  buf_warped_v;
    delete[]  buf_warped_dx;
    delete[]  buf_warped_dy;
    delete[]  buf_warped_residual;
    delete[]  buf_warped_weight;
    delete[]  buf_warped_refColor;
}

/**
 * [CoarseTracker::makeK description]
 * @param HCalib [description]
 * 设置内参
 */
void CoarseTracker::makeK(CalibHessian* HCalib)
{
    w[0] = wG[0];
    h[0] = hG[0];

    fx[0] = HCalib->fxl();
    fy[0] = HCalib->fyl();
    cx[0] = HCalib->cxl();
    cy[0] = HCalib->cyl();

    for (int level = 1; level < pyrLevelsUsed; ++ level)
    {
        w[level] = w[0] >> level;
        h[level] = h[0] >> level;
        fx[level] = fx[level - 1] * 0.5;
        fy[level] = fy[level - 1] * 0.5;
        cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
        cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
    }

    for (int level = 0; level < pyrLevelsUsed; ++ level)
    {
        K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
        Ki[level] = K[level].inverse();
        fxi[level] = Ki[level](0, 0);
        fyi[level] = Ki[level](1, 1);
        cxi[level] = Ki[level](0, 2);
        cyi[level] = Ki[level](1, 2);
    }
}

/**
 * [CoarseTracker::makeCoarseDepthForFirstFrame description]
 * @param fh [description]
 */
void CoarseTracker::makeCoarseDepthForFirstFrame(FrameHessian* fh)
{
    // make coarse tracking templates for latstRef.
    // 分配逆深度图和权重
    memset(idepth[0], 0, sizeof(float)*w[0]*h[0]);
    memset(weightSums[0], 0, sizeof(float)*w[0]*h[0]);

    //遍历每一个点
    for (PointHessian* ph : fh->pointHessians)
    {
        //获取每一个点的坐标
        int u = ph->u + 0.5f;
        int v = ph->v + 0.5f;
        //点的新的逆深度
        float new_idepth = ph->idepth;
        //点的权重
        float weight = sqrtf(1e-3 / (ph->efPoint->HdiF + 1e-12));

        //新的逆深度
        idepth[0][u + w[0]*v] += new_idepth * weight;

        //权重总和
        weightSums[0][u + w[0]*v] += weight;
    }

    //遍历每一层金字塔，从最下层更新最上层的逆深度和权重
    for (int lvl = 1; lvl < pyrLevelsUsed; lvl++)
    {
        // 下一层
        int lvlm1 = lvl - 1;
        int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

        float* idepth_l = idepth[lvl];
        float* weightSums_l = weightSums[lvl];

        float* idepth_lm = idepth[lvlm1];
        float* weightSums_lm = weightSums[lvlm1];

        for (int y = 0; y < hl; y++)
            for (int x = 0; x < wl; x++)
            {
                int bidx = 2 * x   + 2 * y * wlm1;
                idepth_l[x + y * wl] =        idepth_lm[bidx] +
                                              idepth_lm[bidx + 1] +
                                              idepth_lm[bidx + wlm1] +
                                              idepth_lm[bidx + wlm1 + 1];

                weightSums_l[x + y * wl] =    weightSums_lm[bidx] +
                                              weightSums_lm[bidx + 1] +
                                              weightSums_lm[bidx + wlm1] +
                                              weightSums_lm[bidx + wlm1 + 1];
            }
    }

    // dilate idepth by 1.
    // 最下面两层的
    for (int lvl = 0; lvl < 2; lvl++)
    {
        //
        int numIts = 1;

        //
        for (int it = 0; it < numIts; it++)
        {
            int wh = w[lvl] * h[lvl] - w[lvl];
            int wl = w[lvl];
            //当前层的每个点权重
            float* weightSumsl = weightSums[lvl];
            //之前的每个点权重
            float* weightSumsl_bak = weightSums_bak[lvl];
            //weightSumsl替换weightSumsl_bak
            memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
            //当前层的逆深度
            float* idepthl = idepth[lvl];   // dont need to make a temp copy of depth, since I only
            // read values with weightSumsl>0, and write ones with weightSumsl<=0.
            for (int i = w[lvl]; i < wh; i++)
            {
                //若权重小与0或者无权重
                if (weightSumsl_bak[i] <= 0)
                {
                    float sum = 0, num = 0, numn = 0;
                    //寻找四周，四周点的距离更大的权重，根据周围的算当前点的逆深度和权重
                    if (weightSumsl_bak[i + 1 + wl] > 0) { sum += idepthl[i + 1 + wl]; num += weightSumsl_bak[i + 1 + wl]; numn++;}
                    if (weightSumsl_bak[i - 1 - wl] > 0) { sum += idepthl[i - 1 - wl]; num += weightSumsl_bak[i - 1 - wl]; numn++;}
                    if (weightSumsl_bak[i + wl - 1] > 0) { sum += idepthl[i + wl - 1]; num += weightSumsl_bak[i + wl - 1]; numn++;}
                    if (weightSumsl_bak[i - wl + 1] > 0) { sum += idepthl[i - wl + 1]; num += weightSumsl_bak[i - wl + 1]; numn++;}
                    if (numn > 0)
                    {
                        idepthl[i] = sum / numn;
                        weightSumsl[i] = num / numn;
                    }
                }
            }
        }
    }

    // dilate idepth by 1 (2 on lower levels).
    // 第三层到最上层的
    for (int lvl = 2; lvl < pyrLevelsUsed; lvl++)
    {
        int wh = w[lvl] * h[lvl] - w[lvl];
        int wl = w[lvl];
        float* weightSumsl = weightSums[lvl];
        float* weightSumsl_bak = weightSums_bak[lvl];
        memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
        float* idepthl = idepth[lvl];   // dotnt need to make a temp copy of depth, since I only
        // read values with weightSumsl>0, and write ones with weightSumsl<=0.
        for (int i = w[lvl]; i < wh; i++)
        {
            if (weightSumsl_bak[i] <= 0)
            {
                float sum = 0, num = 0, numn = 0;
                if (weightSumsl_bak[i + 1] > 0) { sum += idepthl[i + 1]; num += weightSumsl_bak[i + 1]; numn++;}
                if (weightSumsl_bak[i - 1] > 0) { sum += idepthl[i - 1]; num += weightSumsl_bak[i - 1]; numn++;}
                if (weightSumsl_bak[i + wl] > 0) { sum += idepthl[i + wl]; num += weightSumsl_bak[i + wl]; numn++;}
                if (weightSumsl_bak[i - wl] > 0) { sum += idepthl[i - wl]; num += weightSumsl_bak[i - wl]; numn++;}
                if (numn > 0) {idepthl[i] = sum / numn; weightSumsl[i] = num / numn;}
            }
        }
    }

    // normalize idepths and weights.
    // 归一化每一层的逆深度和权重
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
    {
        float* weightSumsl = weightSums[lvl];
        float* idepthl = idepth[lvl];

        //获取主导帧的灰度图和xy梯度
        Eigen::Vector3f* dIRefl = lastRef->dIp[lvl];

        int wl = w[lvl], hl = h[lvl];

        int lpc_n = 0;
        float* lpc_u = pc_u[lvl];
        float* lpc_v = pc_v[lvl];
        float* lpc_idepth = pc_idepth[lvl];
        float* lpc_color = pc_color[lvl];

        for (int y = 2; y < hl - 2; y++)
            for (int x = 2; x < wl - 2; x++)
            {
                int i = x + y * wl;

                //这个点权重大于0
                if (weightSumsl[i] > 0)
                {
                    //归一化后的逆深度
                    idepthl[i] /= weightSumsl[i];
                    //当前点图像坐标
                    lpc_u[lpc_n] = x;
                    lpc_v[lpc_n] = y;
                    //当前点逆深度
                    lpc_idepth[lpc_n] = idepthl[i];
                    //当前点灰度值
                    lpc_color[lpc_n] = dIRefl[i][0];

                    if (!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i] > 0))
                    {
                        idepthl[i] = -1;
                        continue;   // just skip if something is wrong.
                    }
                    lpc_n++;
                }
                else
                    idepthl[i] = -1;

                weightSumsl[i] = 1;
            }

        //当前层有效的点
        pc_n[lvl] = lpc_n;
//      printf("pc_n[lvl] is %d \n", lpc_n);
    }
}

// make depth mainly from static stereo matching and fill the holes from propogation idpeth map.
/**
 * [CoarseTracker::makeCoarseDepthL0 description]
 * @param frameHessians [description]
 * @param fh_right      [description]
 * @param Hcalib        [description]
 */
void CoarseTracker::makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians, FrameHessian* fh_right, CalibHessian Hcalib)
{
    // make coarse tracking templates for latstRef.
    memset(idepth[0], 0, sizeof(float)*w[0]*h[0]);
    memset(weightSums[0], 0, sizeof(float)*w[0]*h[0]);

    //目标帧
    FrameHessian* fh_target = frameHessians.back();
    //内参
    Mat33f K1 = Mat33f::Identity();
    K1(0, 0) = Hcalib.fxl();
    K1(1, 1) = Hcalib.fyl();
    K1(0, 2) = Hcalib.cxl();
    K1(1, 2) = Hcalib.cyl();

    //遍历每一个关键帧
    for (FrameHessian* fh : frameHessians)
    {
        //遍历这个关键帧中的每一个点
        for (PointHessian* ph : fh->pointHessians)
        {
            if (ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN) //contains information about residuals to the last two (!) frames. ([0] = latest, [1] = the one before).
            {
                PointFrameResidual* r = ph->lastResiduals[0].first;
                assert(r->efResidual->isActive() && r->target == lastRef);
                int u = r->centerProjectedTo[0] + 0.5f;
                int v = r->centerProjectedTo[1] + 0.5f;

                ImmaturePoint* pt_track = new ImmaturePoint((float)u, (float)v, fh_target, &Hcalib);

                pt_track->u_stereo = pt_track->u;
                pt_track->v_stereo = pt_track->v;

                // free to debug
                pt_track->idepth_min_stereo = r->centerProjectedTo[2] * 0.1f;
                pt_track->idepth_max_stereo = r->centerProjectedTo[2] * 1.9f;

                ImmaturePointStatus pt_track_right = pt_track->traceStereo(fh_right, K1, 1);

                float new_idepth = 0;

                if (pt_track_right == ImmaturePointStatus::IPS_GOOD)
                {
                    ImmaturePoint* pt_track_back = new ImmaturePoint(pt_track->lastTraceUV(0), pt_track->lastTraceUV(1), fh_right, &Hcalib);
                    pt_track_back->u_stereo = pt_track_back->u;
                    pt_track_back->v_stereo = pt_track_back->v;


                    pt_track_back->idepth_min_stereo = r->centerProjectedTo[2] * 0.1f;
                    pt_track_back->idepth_max_stereo = r->centerProjectedTo[2] * 1.9f;

                    ImmaturePointStatus pt_track_left = pt_track_back->traceStereo(fh_target, K1, 0);

                    float depth = 1.0f / pt_track->idepth_stereo;
                    float u_delta = abs(pt_track->u - pt_track_back->lastTraceUV(0));
                    if (u_delta < 1 && depth > 0 && depth < 50)
                    {
                        new_idepth = pt_track->idepth_stereo;
                        delete pt_track;
                        delete pt_track_back;

                    }
                    else
                    {
                        new_idepth = r->centerProjectedTo[2];
                        delete pt_track;
                        delete pt_track_back;
                    }

                }
                else {
                    new_idepth = r->centerProjectedTo[2];
                    delete pt_track;
                }
                float weight = sqrtf(1e-3 / (ph->efPoint->HdiF + 1e-12));

                idepth[0][u + w[0]*v] += new_idepth * weight;
                weightSums[0][u + w[0]*v] += weight;

            }
        }
    }

    //从最下层递归最上层的逆深度和权重
    for (int lvl = 1; lvl < pyrLevelsUsed; lvl++)
    {
        int lvlm1 = lvl - 1;
        int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

        float* idepth_l = idepth[lvl];
        float* weightSums_l = weightSums[lvl];

        float* idepth_lm = idepth[lvlm1];
        float* weightSums_lm = weightSums[lvlm1];

        for (int y = 0; y < hl; y++)
            for (int x = 0; x < wl; x++)
            {
                int bidx = 2 * x   + 2 * y * wlm1;
                idepth_l[x + y * wl] =        idepth_lm[bidx] +
                                              idepth_lm[bidx + 1] +
                                              idepth_lm[bidx + wlm1] +
                                              idepth_lm[bidx + wlm1 + 1];

                weightSums_l[x + y * wl] =    weightSums_lm[bidx] +
                                              weightSums_lm[bidx + 1] +
                                              weightSums_lm[bidx + wlm1] +
                                              weightSums_lm[bidx + wlm1 + 1];
            }
    }

    // dilate idepth by 1.
    // 膨胀
    for (int lvl = 0; lvl < 2; lvl++)
    {
        int numIts = 1;

        for (int it = 0; it < numIts; it++)
        {
            int wh = w[lvl] * h[lvl] - w[lvl];
            int wl = w[lvl];
            float* weightSumsl = weightSums[lvl];
            float* weightSumsl_bak = weightSums_bak[lvl];
            memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
            float* idepthl = idepth[lvl];   // dont need to make a temp copy of depth, since I only
            // read values with weightSumsl>0, and write ones with weightSumsl<=0.
            for (int i = w[lvl]; i < wh; i++)
            {
                if (weightSumsl_bak[i] <= 0)
                {
                    float sum = 0, num = 0, numn = 0;
                    if (weightSumsl_bak[i + 1 + wl] > 0) { sum += idepthl[i + 1 + wl]; num += weightSumsl_bak[i + 1 + wl]; numn++;}
                    if (weightSumsl_bak[i - 1 - wl] > 0) { sum += idepthl[i - 1 - wl]; num += weightSumsl_bak[i - 1 - wl]; numn++;}
                    if (weightSumsl_bak[i + wl - 1] > 0) { sum += idepthl[i + wl - 1]; num += weightSumsl_bak[i + wl - 1]; numn++;}
                    if (weightSumsl_bak[i - wl + 1] > 0) { sum += idepthl[i - wl + 1]; num += weightSumsl_bak[i - wl + 1]; numn++;}
                    if (numn > 0) {idepthl[i] = sum / numn; weightSumsl[i] = num / numn;}
                }
            }
        }
    }

    // dilate idepth by 1 (2 on lower levels).
    //膨胀
    for (int lvl = 2; lvl < pyrLevelsUsed; lvl++)
    {
        int wh = w[lvl] * h[lvl] - w[lvl];
        int wl = w[lvl];
        float* weightSumsl = weightSums[lvl];
        float* weightSumsl_bak = weightSums_bak[lvl];
        memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
        float* idepthl = idepth[lvl];   // dotnt need to make a temp copy of depth, since I only
        // read values with weightSumsl>0, and write ones with weightSumsl<=0.
        for (int i = w[lvl]; i < wh; i++)
        {
            if (weightSumsl_bak[i] <= 0)
            {
                float sum = 0, num = 0, numn = 0;
                if (weightSumsl_bak[i + 1] > 0) { sum += idepthl[i + 1]; num += weightSumsl_bak[i + 1]; numn++;}
                if (weightSumsl_bak[i - 1] > 0) { sum += idepthl[i - 1]; num += weightSumsl_bak[i - 1]; numn++;}
                if (weightSumsl_bak[i + wl] > 0) { sum += idepthl[i + wl]; num += weightSumsl_bak[i + wl]; numn++;}
                if (weightSumsl_bak[i - wl] > 0) { sum += idepthl[i - wl]; num += weightSumsl_bak[i - wl]; numn++;}
                if (numn > 0) {idepthl[i] = sum / numn; weightSumsl[i] = num / numn;}
            }
        }
    }

    // normalize idepths and weights.
    //归一化逆深度和权重
    for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
    {
        float* weightSumsl = weightSums[lvl];
        float* idepthl = idepth[lvl];
        Eigen::Vector3f* dIRefl = lastRef->dIp[lvl];

        int wl = w[lvl], hl = h[lvl];

        int lpc_n = 0;
        float* lpc_u = pc_u[lvl];
        float* lpc_v = pc_v[lvl];
        float* lpc_idepth = pc_idepth[lvl];
        float* lpc_color = pc_color[lvl];

        for (int y = 2; y < hl - 2; y++)
            for (int x = 2; x < wl - 2; x++)
            {
                int i = x + y * wl;

                if (weightSumsl[i] > 0)
                {
                    idepthl[i] /= weightSumsl[i];
                    lpc_u[lpc_n] = x;
                    lpc_v[lpc_n] = y;
                    lpc_idepth[lpc_n] = idepthl[i];
                    lpc_color[lpc_n] = dIRefl[i][0];

                    if (!std::isfinite(lpc_color[lpc_n]) || !(idepthl[i] > 0))
                    {
                        idepthl[i] = -1;
                        continue;   // just skip if something is wrong.
                    }
                    lpc_n++;
                }
                else
                    idepthl[i] = -1;

                weightSumsl[i] = 1;
            }

        pc_n[lvl] = lpc_n;
    }
}

/**
 * [CoarseTracker::calcGSSSE description]
 * @param lvl      [description]
 * @param H_out    [description]
 * @param b_out    [description]
 * @param refToNew [description]
 * @param aff_g2l  [description]
 */
void CoarseTracker::calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, SE3 refToNew, AffLight aff_g2l)
{
    //初始化
    acc.initialize();

    //内参
    __m128 fxl = _mm_set1_ps(fx[lvl]);
    __m128 fyl = _mm_set1_ps(fy[lvl]);
    //参考帧b
    __m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
    //两帧变换的a
    __m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]));

    //1
    __m128 one = _mm_set1_ps(1);
    //-1
    __m128 minusOne = _mm_set1_ps(-1);
    //0
    __m128 zero = _mm_set1_ps(0);

    //残差符合阈值的点个数
    int n = buf_warped_n;
    assert(n % 4 == 0);
    for (int i = 0; i < n; i += 4)
    {
        //该点x梯度值
        __m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx + i), fxl);
        //该点y梯度值
        __m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy + i), fyl);
        //该点坐标
        __m128 u = _mm_load_ps(buf_warped_u + i);
        __m128 v = _mm_load_ps(buf_warped_v + i);
        //该点逆深度
        __m128 id = _mm_load_ps(buf_warped_idepth + i);

        //
        acc.updateSSE_eighted(
            _mm_mul_ps(id, dx),
            _mm_mul_ps(id, dy),
            _mm_sub_ps(zero, _mm_mul_ps(id, _mm_add_ps(_mm_mul_ps(u, dx), _mm_mul_ps(v, dy)))),
            _mm_sub_ps(zero, _mm_add_ps(
                           _mm_mul_ps(_mm_mul_ps(u, v), dx),
                           _mm_mul_ps(dy, _mm_add_ps(one, _mm_mul_ps(v, v))))),
            _mm_add_ps(
                _mm_mul_ps(_mm_mul_ps(u, v), dy),
                _mm_mul_ps(dx, _mm_add_ps(one, _mm_mul_ps(u, u)))),
            _mm_sub_ps(_mm_mul_ps(u, dy), _mm_mul_ps(v, dx)),
            _mm_mul_ps(a, _mm_sub_ps(b0, _mm_load_ps(buf_warped_refColor + i))),
            minusOne,
            _mm_load_ps(buf_warped_residual + i),
            _mm_load_ps(buf_warped_weight + i));
    }

    //
    acc.finish();

    //左上角8*8矩阵/点个数为H
    H_out = acc.H.topLeftCorner<8, 8>().cast<double>() * (1.0f / n);
    //右上角8*1矩阵/点个数
    b_out = acc.H.topRightCorner<8, 1>().cast<double>() * (1.0f / n);

    //乘以一个比例
    H_out.block<8, 3>(0, 0) *= SCALE_XI_ROT;
    H_out.block<8, 3>(0, 3) *= SCALE_XI_TRANS;
    H_out.block<8, 1>(0, 6) *= SCALE_A;
    H_out.block<8, 1>(0, 7) *= SCALE_B;
    H_out.block<3, 8>(0, 0) *= SCALE_XI_ROT;
    H_out.block<3, 8>(3, 0) *= SCALE_XI_TRANS;
    H_out.block<1, 8>(6, 0) *= SCALE_A;
    H_out.block<1, 8>(7, 0) *= SCALE_B;

    b_out.segment<3>(0) *= SCALE_XI_ROT;
    b_out.segment<3>(3) *= SCALE_XI_TRANS;
    b_out.segment<1>(6) *= SCALE_A;
    b_out.segment<1>(7) *= SCALE_B;
}

/**
 * [CoarseTracker::calcRes description]
 * @param  lvl      [description]
 * @param  refToNew [description]
 * @param  aff_g2l  [description]
 * @param  cutoffTH [description]
 * @return          [description]
 */
Vec6 CoarseTracker::calcRes(int lvl, SE3 refToNew, AffLight aff_g2l, float cutoffTH)
{
    //当前能量值
    float E = 0;
    //总共点个数
    int numTermsInE = 0;
    //变换后的点个数
    int numTermsInWarped = 0;

    //误差过大的点个数
    int numSaturated = 0;

    //当前层图像大小
    int wl = w[lvl];
    int hl = h[lvl];
    //新一帧的灰度和梯度值
    Eigen::Vector3f* dINewl = newFrame->dIp[lvl]; //先粗糙估计
    float fxl = fx[lvl];
    float fyl = fy[lvl];
    float cxl = cx[lvl];
    float cyl = cy[lvl];

    //R*K'
    Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
    // printf("the Ki is:\n %f,%f,%f\n %f,%f,%f\n %f,%f,%f\n -----\n",Ki[lvl](0,0), Ki[lvl](0,1), Ki[lvl](0,2), Ki[lvl](1,0), Ki[lvl](1,1), Ki[lvl](1,2), Ki[lvl](2,0), Ki[lvl](2,1), Ki[lvl](2,2) );
    //t
    Vec3f t = (refToNew.translation()).cast<float>();
    // printf("the t is:\n %f, %f, %f\n", t(0),t(1),t(2));
    //a和b的变换
    Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l).cast<float>();

    //总共T,RT,num
    float sumSquaredShiftT = 0;
    float sumSquaredShiftRT = 0;
    float sumSquaredShiftNum = 0;

    //最大能量值
    float maxEnergy = 2 * setting_huberTH * cutoffTH - setting_huberTH * setting_huberTH; // energy for r=setting_coarseCutoffTH.

    //残差图
    MinimalImageB3* resImage = 0;
    if (debugPlot)
    {
        resImage = new MinimalImageB3(wl, hl);
        resImage->setConst(Vec3b(255, 255, 255));
    }

    //当前层有效点个数
    int nl = pc_n[lvl];
    //当前层点图像坐标
    float* lpc_u = pc_u[lvl];
    float* lpc_v = pc_v[lvl];
    //当前层逆深度
    float* lpc_idepth = pc_idepth[lvl];
    //当前层灰度值
    float* lpc_color = pc_color[lvl];

//  printf("the num of the points is: %d \n", nl);
    for (int i = 0; i < nl; i++)
    {
        float id = lpc_idepth[i];
        float x = lpc_u[i];
        float y = lpc_v[i];

        //重投影后
        Vec3f pt = RKi * Vec3f(x, y, 1) + t * id;
        float u = pt[0] / pt[2];
        float v = pt[1] / pt[2];
        float Ku = fxl * u + cxl;
        float Kv = fyl * v + cyl;
        float new_idepth = id / pt[2];
        // printf("Ku & Kv are: %f, %f; x and y are: %f, %f\n", Ku, Kv, x, y);

        //第一层并且i是32的倍数
        if (lvl == 0 && i % 32 == 0)
        {
            // translation only (positive)
            //只进行正平移
            Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t * id;
            float uT = ptT[0] / ptT[2];
            float vT = ptT[1] / ptT[2];
            float KuT = fxl * uT + cxl;
            float KvT = fyl * vT + cyl;

            // translation only (negative)
            // 只进行负平移
            Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t * id;
            float uT2 = ptT2[0] / ptT2[2];
            float vT2 = ptT2[1] / ptT2[2];
            float KuT2 = fxl * uT2 + cxl;
            float KvT2 = fyl * vT2 + cyl;

            //translation and rotation (negative)
            //旋转+只进行正平移
            Vec3f pt3 = RKi * Vec3f(x, y, 1) - t * id;
            float u3 = pt3[0] / pt3[2];
            float v3 = pt3[1] / pt3[2];
            float Ku3 = fxl * u3 + cxl;
            float Kv3 = fyl * v3 + cyl;

            //translation and rotation (positive)
            //already have it.

            //几种变换后的像素坐标偏差和
            sumSquaredShiftT += (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
            sumSquaredShiftT += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
            sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
            sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
            sumSquaredShiftNum += 2;
        }

        if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0))
            continue;

        //参考帧这个点的灰度值
        float refColor = lpc_color[i];
        //当前帧改点重投影后线性差值的
        Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
        if (!std::isfinite((float)hitColor[0]))
            continue;

        //残差
        float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
        //Huber weight
        //huber权重
        float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

        //残差大于阈值，残差过大
        if (fabs(residual) > cutoffTH)
        {
            //设置该点
            if (debugPlot)
                resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0, 0, 255));

            //直接加最大能量
            E += maxEnergy;
            //个数numTermsInE++
            numTermsInE++;
            numSaturated++;
        }
        else
        {
            if (debugPlot)
                resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(residual + 128, residual + 128, residual + 128));

            //huber残差
            E += hw * residual * residual * (2 - hw);
            //个数numTermsInE++
            numTermsInE++;

            //变换后的点的
            buf_warped_idepth[numTermsInWarped] = new_idepth;
            buf_warped_u[numTermsInWarped] = u;
            buf_warped_v[numTermsInWarped] = v;
            buf_warped_dx[numTermsInWarped] = hitColor[1];
            buf_warped_dy[numTermsInWarped] = hitColor[2];
            buf_warped_residual[numTermsInWarped] = residual;
            buf_warped_weight[numTermsInWarped] = hw;
            buf_warped_refColor[numTermsInWarped] = lpc_color[i];

            //变换后的点数++
            numTermsInWarped++;
        }
    }

    //如过变换成功的点个数是不是4的倍数，则跳过
    while (numTermsInWarped % 4 != 0)
    {
        buf_warped_idepth[numTermsInWarped] = 0;
        buf_warped_u[numTermsInWarped] = 0;
        buf_warped_v[numTermsInWarped] = 0;
        buf_warped_dx[numTermsInWarped] = 0;
        buf_warped_dy[numTermsInWarped] = 0;
        buf_warped_residual[numTermsInWarped] = 0;
        buf_warped_weight[numTermsInWarped] = 0;
        buf_warped_refColor[numTermsInWarped] = 0;
        numTermsInWarped++;
    }
    //总共残差符合期望的变换的个数
    buf_warped_n = numTermsInWarped;

    //显示残差图
    if (debugPlot)
    {
        IOWrap::displayImage("RES", resImage, false);
        IOWrap::waitKey(0);
        delete resImage;
    }

    Vec6 rs;
    //总误差能量值
    rs[0] = E;
    //总共点的个数
    rs[1] = numTermsInE;
    //平移后的像素重投影误差误差/个数/2
    rs[2] = sumSquaredShiftT / (sumSquaredShiftNum + 0.1);
    //
    rs[3] = 0;
    //平移旋转后像素重投影误差误差/个数/2
    rs[4] = sumSquaredShiftRT / (sumSquaredShiftNum + 0.1);
    //误差过大的点比例
    rs[5] = numSaturated / (float)numTermsInE;

    return rs;
}

/**
 * [CoarseTracker::setCTRefForFirstFrame description]
 * @param frameHessians [description]
 */
void CoarseTracker::setCTRefForFirstFrame(std::vector<FrameHessian *> frameHessians)
{
    assert(frameHessians.size() > 0);
    //获取参考帧
    lastRef = frameHessians.back();

    //计算当前参考帧的深度图
    makeCoarseDepthForFirstFrame(lastRef);

    //获取参考帧的id
    refFrameID = lastRef->shell->id;
    //获取参考帧的a和b
    lastRef_aff_g2l = lastRef->aff_g2l();

    //初始RMSE
    firstCoarseRMSE = -1;
}

/**
 * [CoarseTracker::setCoarseTrackingRef description]
 * @param frameHessians [description]
 * @param fh_right      [description]
 * @param Hcalib        [description]
 */
void CoarseTracker::setCoarseTrackingRef(
    std::vector<FrameHessian*> frameHessians, FrameHessian* fh_right, CalibHessian Hcalib)
{
    assert(frameHessians.size() > 0);
    //参考帧
    lastRef = frameHessians.back();

    //计算当前参考帧的深度图
    makeCoarseDepthL0(frameHessians, fh_right, Hcalib);

    //参考帧id
    refFrameID = lastRef->shell->id;
    lastRef_aff_g2l = lastRef->aff_g2l();

    //初始的RMSE
    firstCoarseRMSE = -1;
}

/**
 * [CoarseTracker::trackNewestCoarse description]
 * @param  newFrameHessian [description]
 * @param  lastToNew_out   [description]
 * @param  aff_g2l_out     [description]
 * @param  coarsestLvl     [description]
 * @param  minResForAbort  [description]
 * @param  wrap            [description]
 * @return                 [description]
 */
bool CoarseTracker::trackNewestCoarse(
    FrameHessian* newFrameHessian,
    SE3 &lastToNew_out, AffLight &aff_g2l_out,
    int coarsestLvl,
    Vec5 minResForAbort,
    IOWrap::Output3DWrapper* wrap)
{
    debugPlot = setting_render_displayCoarseTrackingFull;
    debugPrint = false;

    assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

    //设置最新的残差
    lastResiduals.setConstant(NAN);
    //设置最新的
    lastFlowIndicators.setConstant(1000);

    //新一帧
    newFrame = newFrameHessian;
    //每一层最大迭代次数
    int maxIterations[] = {10, 20, 50, 50, 50};
    //lambda
    float lambdaExtrapolationLimit = 0.001;

    //当前的位姿和a,b
    SE3 refToNew_current = lastToNew_out;
    AffLight aff_g2l_current = aff_g2l_out;

    bool haveRepeated = false;

    //从最上层开始
    for (int lvl = coarsestLvl; lvl >= 0; lvl--)
    {
        //Hessian矩阵，b矩阵
        Mat88 H; Vec8 b;
        float levelCutoffRepeat = 1;
        //计算残差，层号，当前位姿，当前a和b,阈值20*levelCutoffRepeat
        //0：总误差能量值
        //1：总共点的个数
        //2：平移后的像素重投影误差误差/个数/2
        //3： 0
        //4：平移旋转后像素重投影误差误差/个数/2
        //5：误差过大的点比例
        Vec6 resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH * levelCutoffRepeat);

        //若误差多大的比例>0.6且levelCutoffRepeat<50
        while (resOld[5] > 0.6 && levelCutoffRepeat < 50)
        {
            //一直计算,增大阈值
            levelCutoffRepeat *= 2;
            //再计算残差
            resOld = calcRes(lvl, refToNew_current, aff_g2l_current, setting_coarseCutoffTH * levelCutoffRepeat);

            if (!setting_debugout_runquiet)
                printf("INCREASING cutoff to %f (ratio is %f)!\n", setting_coarseCutoffTH * levelCutoffRepeat, resOld[5]);
        }

        //更新当前H和b矩阵
        calcGSSSE(lvl, H, b, refToNew_current, aff_g2l_current);

        //lambda
        float lambda = 0.01;

        if (debugPrint)
        {
            Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_current).cast<float>();
            printf("lvl%d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
                   lvl, -1, lambda, 1.0f,
                   "INITIA",
                   0.0f,
                   resOld[0] / resOld[1],
                   0, (int)resOld[1],
                   0.0f);
            std::cout << refToNew_current.log().transpose() << " AFF " << aff_g2l_current.vec().transpose() << " (rel " << relAff.transpose() << ")\n";
        }

        //迭代
        for (int iteration = 0; iteration < maxIterations[lvl]; iteration++)
        {
            //Hessian矩阵
            Mat88 Hl = H;
            for (int i = 0; i < 8; i++)
                Hl(i, i) *= (1 + lambda);

            //x=H×-b,使用LDLT分解
            Vec8 inc = Hl.ldlt().solve(-b);

            //只更新位姿
            if (setting_affineOptModeA < 0 && setting_affineOptModeB < 0)   // fix a, b
            {
                inc.head<6>() = Hl.topLeftCorner<6, 6>().ldlt().solve(-b.head<6>());
                inc.tail<2>().setZero();
            }
            //只更新位姿和A
            if (!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0) // fix b
            {
                inc.head<7>() = Hl.topLeftCorner<7, 7>().ldlt().solve(-b.head<7>());
                inc.tail<1>().setZero();
            }
            //只更新位姿和B
            if (setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0)) // fix a
            {
                Mat88 HlStitch = Hl;
                Vec8 bStitch = b;
                //第6位为第7
                HlStitch.col(6) = HlStitch.col(7);
                HlStitch.row(6) = HlStitch.row(7);
                bStitch[6] = bStitch[7];
                Vec7 incStitch = HlStitch.topLeftCorner<7, 7>().ldlt().solve(-bStitch.head<7>());
                inc.setZero();
                inc.head<6>() = incStitch.head<6>();
                inc[6] = 0;
                inc[7] = incStitch[6];
            }

            //lambda<lambdaExtrapolationLimit,则0.001/lambda
            float extrapFac = 1;
            if (lambda < lambdaExtrapolationLimit)
                extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));

            //乘以lambda
            inc *= extrapFac;

            Vec8 incScaled = inc;

            //乘以尺度比例
            incScaled.segment<3>(0) *= SCALE_XI_ROT;
            incScaled.segment<3>(3) *= SCALE_XI_TRANS;
            incScaled.segment<1>(6) *= SCALE_A;
            incScaled.segment<1>(7) *= SCALE_B;

            if (!std::isfinite(incScaled.sum()))
                incScaled.setZero();

            //最新的位姿和光度参数
            SE3 refToNew_new = SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;
            AffLight aff_g2l_new = aff_g2l_current;
            aff_g2l_new.a += incScaled[6];
            aff_g2l_new.b += incScaled[7];

            //再计算残差，计算变换后的点
            Vec6 resNew = calcRes(lvl, refToNew_new, aff_g2l_new, setting_coarseCutoffTH * levelCutoffRepeat);

            //残差是否减小
            bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

            if (debugPrint)
            {
                Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_new).cast<float>();
                printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
                       lvl, iteration, lambda,
                       extrapFac,
                       (accept ? "ACCEPT" : "REJECT"),
                       resOld[0] / resOld[1],
                       resNew[0] / resNew[1],
                       (int)resOld[1], (int)resNew[1],
                       inc.norm());
                std::cout << refToNew_new.log().transpose() << " AFF " << aff_g2l_new.vec().transpose() << " (rel " << relAff.transpose() << ")\n";
            }
            if (accept)
            {
                //更新当前H和b矩阵
                calcGSSSE(lvl, H, b, refToNew_new, aff_g2l_new);
                //更新残差和a和b和位姿
                resOld = resNew;
                // printf("accepted with res: %f\n", resOld[0]/resOld[1]);
                aff_g2l_current = aff_g2l_new;
                refToNew_current = refToNew_new;
                //lambda减小一倍
                lambda *= 0.5;
            }
            else
            {
                //失败，则lambda*4
                lambda *= 4;
                //lambda小于lambdaExtrapolationLimit，
                //lambda等于lambdaExtrapolationLimit
                if (lambda < lambdaExtrapolationLimit)
                    lambda = lambdaExtrapolationLimit;
            }

            //矩阵行列式<1e-3，则说明迭代够小了
            if (!(inc.norm() > 1e-3))
            {
                if (debugPrint)
                    printf("inc too small, break!\n");
                break;
            }
        }

        // set last residual for that level, as well as flow indicators.
        //残差
        lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));
        //从第三位开始后面的三位
        lastFlowIndicators = resOld.segment<3>(2);

        //lastResiduals大于1.5*每一层的阈值，则说明失败了
        if (lastResiduals[lvl] > 1.5 * minResForAbort[lvl])
            return false;

        if (levelCutoffRepeat > 1 && !haveRepeated)
        {
            lvl++;
            haveRepeated = true;
            printf("REPEAT LEVEL!\n");
        }
    }

    // set!
    //设置最新的位姿和a和b
    lastToNew_out = refToNew_current;
    aff_g2l_out = aff_g2l_current;

    //若a和b计算的值超出阈值，则失败
    if ((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 1.2))
            || (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200)))
        return false;

    //相对变换
    Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_out).cast<float>();

    if ((setting_affineOptModeA == 0 && (fabsf(logf((float)relAff[0])) > 1.5))
            || (setting_affineOptModeB == 0 && (fabsf((float)relAff[1]) > 200)))
        return false;

    //不优化a和b，则这两个值=0
    if (setting_affineOptModeA < 0) aff_g2l_out.a = 0;
    if (setting_affineOptModeB < 0) aff_g2l_out.b = 0;

    return true;
}

/**
 * [CoarseTracker::debugPlotIDepthMap description]
 * @param minID_pt [description]
 * @param maxID_pt [description]
 * @param wraps    [description]
 */
void CoarseTracker::debugPlotIDepthMap(float* minID_pt, float* maxID_pt, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if (w[1] == 0) return;
    int lvl = 0;
    {
        std::vector<float> allID;
        for (int i = 0; i < h[lvl]*w[lvl]; i++)
        {
            if (idepth[lvl][i] > 0)
                allID.push_back(idepth[lvl][i]);
        }
        std::sort(allID.begin(), allID.end());
        int n = allID.size() - 1;

        float minID_new = allID[(int)(n * 0.05)];
        float maxID_new = allID[(int)(n * 0.95)];

        float minID, maxID;
        minID = minID_new;
        maxID = maxID_new;
        if (minID_pt != 0 && maxID_pt != 0)
        {
            if (*minID_pt < 0 || *maxID_pt < 0)
            {
                *maxID_pt = maxID;
                *minID_pt = minID;
            }
            else
            {

                // slowly adapt: change by maximum 10% of old span.
                float maxChange = 0.3 * (*maxID_pt - *minID_pt);

                if (minID < *minID_pt - maxChange)
                    minID = *minID_pt - maxChange;
                if (minID > *minID_pt + maxChange)
                    minID = *minID_pt + maxChange;


                if (maxID < *maxID_pt - maxChange)
                    maxID = *maxID_pt - maxChange;
                if (maxID > *maxID_pt + maxChange)
                    maxID = *maxID_pt + maxChange;

                *maxID_pt = maxID;
                *minID_pt = minID;
            }
        }

        MinimalImageB3 mf(w[lvl], h[lvl]);
        mf.setBlack();
        for (int i = 0; i < h[lvl]*w[lvl]; i++)
        {
            int c = lastRef->dIp[lvl][i][0] * 0.9f;
            if (c > 255) c = 255;
            mf.at(i) = Vec3b(c, c, c);
        }
        int wl = w[lvl];
        for (int y = 3; y < h[lvl] - 3; y++)
            for (int x = 3; x < wl - 3; x++)
            {
                int idx = x + y * wl;
                float sid = 0, nid = 0;
                float* bp = idepth[lvl] + idx;

                if (bp[0] > 0) {sid += bp[0]; nid++;}
                if (bp[1] > 0) {sid += bp[1]; nid++;}
                if (bp[-1] > 0) {sid += bp[-1]; nid++;}
                if (bp[wl] > 0) {sid += bp[wl]; nid++;}
                if (bp[-wl] > 0) {sid += bp[-wl]; nid++;}

                if (bp[0] > 0 || nid >= 3)
                {
                    float id = ((sid / nid) - minID) / ((maxID - minID));
                    mf.setPixelCirc(x, y, makeJet3B(id));
                    //mf.at(idx) = makeJet3B(id);
                }
            }
        //IOWrap::displayImage("coarseDepth LVL0", &mf, false);


        for (IOWrap::Output3DWrapper* ow : wraps)
            ow->pushDepthImage(&mf);

        if (debugSaveImages)
        {
            char buf[1000];
            snprintf(buf, 1000, "images_out/predicted_%05d_%05d.png", lastRef->shell->id, refFrameID);
            IOWrap::writeImage(buf, &mf);
        }
    }
}

/**
 * [CoarseTracker::debugPlotIDepthMapFloat description]
 * @param wraps [description]
 */
void CoarseTracker::debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if (w[1] == 0) return;
    int lvl = 0;
    MinimalImageF mim(w[lvl], h[lvl], idepth[lvl]);
    for (IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImageFloat(&mim, lastRef);
}


CoarseDistanceMap::CoarseDistanceMap(int ww, int hh)
{
    fwdWarpedIDDistFinal = new float[ww * hh / 4];

    bfsList1 = new Eigen::Vector2i[ww * hh / 4];
    bfsList2 = new Eigen::Vector2i[ww * hh / 4];

    int fac = 1 << (pyrLevelsUsed - 1);

    coarseProjectionGrid = new PointFrameResidual*[2048 * (ww * hh / (fac * fac))];
    coarseProjectionGridNum = new int[ww * hh / (fac * fac)];

    w[0] = h[0] = 0;
}

/**
 *
 */
CoarseDistanceMap::~CoarseDistanceMap()
{
    delete[] fwdWarpedIDDistFinal;
    delete[] bfsList1;
    delete[] bfsList2;
    delete[] coarseProjectionGrid;
    delete[] coarseProjectionGridNum;
}

/**
 * [CoarseDistanceMap::makeDistanceMap description]
 * @param frameHessians [description]
 * @param frame         [description]
 */
void CoarseDistanceMap::makeDistanceMap(
    std::vector<FrameHessian*> frameHessians,
    FrameHessian* frame)
{
    int w1 = w[1];
    int h1 = h[1];
    int wh1 = w1 * h1;
    for (int i = 0; i < wh1; i++)
        fwdWarpedIDDistFinal[i] = 1000;

    // make coarse tracking templates for latstRef.
    int numItems = 0;

    for (FrameHessian* fh : frameHessians)
    {
        if (frame == fh) continue;

        SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
        Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]);
        Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

        for (PointHessian* ph : fh->pointHessians)
        {
            assert(ph->status == PointHessian::ACTIVE);
            Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt * ph->idepth_scaled;
            int u = ptp[0] / ptp[2] + 0.5f;
            int v = ptp[1] / ptp[2] + 0.5f;
            if (!(u > 0 && v > 0 && u < w[1] && v < h[1])) continue;
            fwdWarpedIDDistFinal[u + w1 * v] = 0;
            bfsList1[numItems] = Eigen::Vector2i(u, v);
            numItems++;
        }
    }

    growDistBFS(numItems);
}

/**
 * [CoarseDistanceMap::makeInlierVotes description]
 * @param frameHessians [description]
 */
void CoarseDistanceMap::makeInlierVotes(std::vector<FrameHessian*> frameHessians)
{

}

/**
 * [CoarseDistanceMap::growDistBFS description]
 * @param bfsNum [description]
 */
void CoarseDistanceMap::growDistBFS(int bfsNum)
{
    assert(w[0] != 0);
    int w1 = w[1], h1 = h[1];
    for (int k = 1; k < 40; k++) // original K is 40
    {
        int bfsNum2 = bfsNum;
        std::swap<Eigen::Vector2i*>(bfsList1, bfsList2);
        bfsNum = 0;

        if (k % 2 == 0)
        {
            for (int i = 0; i < bfsNum2; i++)
            {
                int x = bfsList2[i][0];
                int y = bfsList2[i][1];
                if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
                int idx = x + y * w1;

                if (fwdWarpedIDDistFinal[idx + 1] > k)
                {
                    fwdWarpedIDDistFinal[idx + 1] = k;
                    bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y); bfsNum++;
                }
                if (fwdWarpedIDDistFinal[idx - 1] > k)
                {
                    fwdWarpedIDDistFinal[idx - 1] = k;
                    bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y); bfsNum++;
                }
                if (fwdWarpedIDDistFinal[idx + w1] > k)
                {
                    fwdWarpedIDDistFinal[idx + w1] = k;
                    bfsList1[bfsNum] = Eigen::Vector2i(x, y + 1); bfsNum++;
                }
                if (fwdWarpedIDDistFinal[idx - w1] > k)
                {
                    fwdWarpedIDDistFinal[idx - w1] = k;
                    bfsList1[bfsNum] = Eigen::Vector2i(x, y - 1); bfsNum++;
                }
            }
        }
        else
        {
            for (int i = 0; i < bfsNum2; i++)
            {
                int x = bfsList2[i][0];
                int y = bfsList2[i][1];
                if (x == 0 || y == 0 || x == w1 - 1 || y == h1 - 1) continue;
                int idx = x + y * w1;

                if (fwdWarpedIDDistFinal[idx + 1] > k)
                {
                    fwdWarpedIDDistFinal[idx + 1] = k;
                    bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y); bfsNum++;
                }
                if (fwdWarpedIDDistFinal[idx - 1] > k)
                {
                    fwdWarpedIDDistFinal[idx - 1] = k;
                    bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y); bfsNum++;
                }
                if (fwdWarpedIDDistFinal[idx + w1] > k)
                {
                    fwdWarpedIDDistFinal[idx + w1] = k;
                    bfsList1[bfsNum] = Eigen::Vector2i(x, y + 1); bfsNum++;
                }
                if (fwdWarpedIDDistFinal[idx - w1] > k)
                {
                    fwdWarpedIDDistFinal[idx - w1] = k;
                    bfsList1[bfsNum] = Eigen::Vector2i(x, y - 1); bfsNum++;
                }

                if (fwdWarpedIDDistFinal[idx + 1 + w1] > k)
                {
                    fwdWarpedIDDistFinal[idx + 1 + w1] = k;
                    bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y + 1); bfsNum++;
                }
                if (fwdWarpedIDDistFinal[idx - 1 + w1] > k)
                {
                    fwdWarpedIDDistFinal[idx - 1 + w1] = k;
                    bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y + 1); bfsNum++;
                }
                if (fwdWarpedIDDistFinal[idx - 1 - w1] > k)
                {
                    fwdWarpedIDDistFinal[idx - 1 - w1] = k;
                    bfsList1[bfsNum] = Eigen::Vector2i(x - 1, y - 1); bfsNum++;
                }
                if (fwdWarpedIDDistFinal[idx + 1 - w1] > k)
                {
                    fwdWarpedIDDistFinal[idx + 1 - w1] = k;
                    bfsList1[bfsNum] = Eigen::Vector2i(x + 1, y - 1); bfsNum++;
                }
            }
        }
    }
}

/**
 * [CoarseDistanceMap::addIntoDistFinal description]
 * @param u [description]
 * @param v [description]
 */
void CoarseDistanceMap::addIntoDistFinal(int u, int v)
{
    if (w[0] == 0) return;
    bfsList1[0] = Eigen::Vector2i(u, v);
    fwdWarpedIDDistFinal[u + w[1]*v] = 0;
    growDistBFS(1);
}

/**
 * [CoarseDistanceMap::makeK description]
 * @param HCalib [description]
 */
void CoarseDistanceMap::makeK(CalibHessian* HCalib)
{
    w[0] = wG[0];
    h[0] = hG[0];

    fx[0] = HCalib->fxl();
    fy[0] = HCalib->fyl();
    cx[0] = HCalib->cxl();
    cy[0] = HCalib->cyl();

    for (int level = 1; level < pyrLevelsUsed; ++ level)
    {
        w[level] = w[0] >> level;
        h[level] = h[0] >> level;
        fx[level] = fx[level - 1] * 0.5;
        fy[level] = fy[level - 1] * 0.5;
        cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
        cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
    }

    for (int level = 0; level < pyrLevelsUsed; ++ level)
    {
        K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
        Ki[level] = K[level].inverse();
        fxi[level] = Ki[level](0, 0);
        fyi[level] = Ki[level](1, 1);
        cxi[level] = Ki[level](0, 2);
        cyi[level] = Ki[level](1, 2);
    }
}

}
