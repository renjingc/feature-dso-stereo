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



#include "FullSystem/ImmaturePoint.h"
#include "FullSystem/FrameShell.h"
#include "FullSystem/ResidualProjections.h"

namespace fdso
{

/**
 * @brief      Constructs the object.
 *
 * @param[in]  u_      { parameter_description }
 * @param[in]  v_      { parameter_description }
 * @param      host_   The host
 * @param[in]  type    The type
 * @param      HCalib  The h calib
 * 初始化一个点
 * my_type要么是1，要么是0，好像一直没用到
 */
ImmaturePoint::ImmaturePoint(int u_, int v_, FrameHessian* host_, float type, CalibHessian* HCalib)
	: u(u_), v(v_), host(host_), my_type(type), idepth_min(0), idepth_max(NAN), idepth_min_stereo(0), idepth_max_stereo(NAN), lastTraceStatus(IPS_UNINITIALIZED)
{
	//梯度设为0
	gradH.setZero();  //Mat22f gradH

	//遍历模式
	for (int idx = 0; idx < patternNum; idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u + dx, v + dy, wG[0]);

		//每个模式的灰度值
		color[idx] = ptc[0];
		if (!std::isfinite(color[idx])) {energyTH = NAN; return;}

		//加上每个点的梯度值，２*2矩阵,最后gradH为模式的每个点的梯度矩阵和
		gradH += ptc.tail<2>()  * ptc.tail<2>().transpose();

		//这个点的权值sqrt(c/(c+梯度值))   50*50/(50*50+梯度平方和)
		weights[idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
	}

#if USE_NCC
	//- Get patternNCCHostNormalized from host
	for (int idx = 0; idx < patternNumNCC; idx++) {
		int dx = patternPNCC[idx][0];
		int dy = patternPNCC[idx][1];

		Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u + dx, v + dy, wG[0]);


		patternNCCHostNormalized[idx] = ptc[0];
	}
	patternNCCHostNormalized.normalize();
#endif


	//设置outlier阈值，这里设为固定值，在迭代计算点逆深度和最佳重投影坐标时，用到这个阈值判断误差能量阈值
	energyTH = patternNum * setting_outlierTH;
	//energyTH*=1
	energyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;

	//质量
	quality = 10000;

	idepth_stereo = 0.0;
}

/**
 * @brief      Constructs the object.
 *
 * @param[in]  u_      { parameter_description }
 * @param[in]  v_      { parameter_description }
 * @param      host_   The host
 * @param      HCalib  The h calib
 */
ImmaturePoint::ImmaturePoint(float u_, float v_, FrameHessian* host_, CalibHessian* HCalib)
	: u(u_), v(v_), host(host_), idepth_min(0), idepth_max(NAN), idepth_min_stereo(0), idepth_max_stereo(NAN), lastTraceStatus(IPS_UNINITIALIZED)
{
	//梯度设为0
	gradH.setZero();  //Mat22f gradH

	//遍历模式
	for (int idx = 0; idx < patternNum; idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u + dx, v + dy, wG[0]);

		color[idx] = ptc[0];
		if (!std::isfinite(color[idx])) {energyTH = NAN; return;}

		//加上每个点的梯度值，２*2矩阵,最后gradH为模式的每个点的梯度矩阵和
		gradH += ptc.tail<2>()  * ptc.tail<2>().transpose();

		//这个点的权值sqrt(c/(c+梯度值))
		weights[idx] = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + ptc.tail<2>().squaredNorm()));
	}

#if USE_NCC
	//- Get patternNCCHostNormalized from host
	for (int idx = 0; idx < patternNumNCC; idx++) {
		int dx = patternPNCC[idx][0];
		int dy = patternPNCC[idx][1];

		Vec3f ptc = getInterpolatedElement33BiLin(host->dI, u + dx, v + dy, wG[0]);


		patternNCCHostNormalized[idx] = ptc[0];
	}
	patternNCCHostNormalized.normalize();
#endif

	//设置outlier阈值，这里设为固定值，在迭代计算点逆深度和最佳重投影坐标时，用到这个阈值判断误差能量阈值
	energyTH = patternNum * setting_outlierTH;
	//energyTH*=1
	energyTH *= setting_overallEnergyTHWeight * setting_overallEnergyTHWeight;
	//质量
	quality = 10000;
}

ImmaturePoint::~ImmaturePoint()
{
	if (mF)
		mF = nullptr;
}
// do static stereo match. if mode_right = true, it matches from left to right. otherwise do it from right to left.


#if USE_NCC

// modeRight == true, from left to right, modeRight == false, from right to left
ImmaturePointStatus ImmaturePoint::traceStereo(FrameHessian* frameRight, Mat33f K, bool modeRight) 
{
	// KRKi
	Mat33f KRKi = Mat33f::Identity().cast<float>();
	// Kt
	Vec3f Kt;
	// T between stereo cameras
	Vec3f bl;

	if (modeRight)
		bl << -baseline, 0, 0;
	else
		bl << baseline, 0, 0;


	Kt = K * bl;
	Vec2f aff;
	aff << 1, 0;

	// baseline * fx
	float bf = -K(0, 0) * bl[0];

	Vec3f pr = KRKi * Vec3f(u_stereo, v_stereo, 1);
	Vec3f ptpMin = pr + Kt * idepth_min_stereo;

	float uMin = ptpMin[0] / ptpMin[2];
	float vMin = ptpMin[1] / ptpMin[2];

	//- Check if out of boundary
	if (!(uMin > 4 && vMin > 4 && uMin < wG[0] - 5 && vMin < hG[0] - 5)) {
		lastTraceUV = Vec2f(-1, -1);
		lastTracePixelInterval = 0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

	float dist;
	float uMax;
	float vMax;
	Vec3f ptpMax;
	//- setting_maxPixSearch = 0.027
	//- maxPixSearch is the searched epipolar line length in pixel. Of course in pixel :).
	float maxPixSearch = (wG[0] + hG[0]) * setting_maxPixSearch;

	if (std::isfinite(idepth_max_stereo)) { //- If the min depth is finite (no too close), calculate the normal dist.
		ptpMax = pr + Kt * idepth_max_stereo;
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];


		if (!(uMax > 4 && vMax > 4 && uMax < wG[0] - 5 && vMax < hG[0] - 5)) {
			lastTraceUV = Vec2f(-1, -1);
			lastTracePixelInterval = 0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}

		// ============== check their distance. everything below 2px is OK (-> skip). ===================
		//- If the depth range too small, no need to further optimize the depth.
		dist = sqrtf((uMin - uMax) * (uMin - uMax) + (vMin - vMax) * (vMin - vMax));
		if (dist < setting_trace_slackInterval) //- setting_trace_slackInterval = 1.5
			return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
		assert(dist > 0);
	}
	else {
		dist = maxPixSearch;

		// project to arbitrary depth to get direction.
		ptpMax = pr + Kt * 0.01; // 0.01 has not meanning here, calculate ptpMax just for a diretion
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];

		// direction.
		float dx = uMax - uMin;
		float dy = vMax - vMin;
		float d = 1.0f / sqrtf(dx * dx + dy * dy);

		// set to [setting_maxPixSearch].
		uMax = uMin + dist * (dx * d);
		vMax = vMin + dist * (dy * d);

		//- Check if out of boundary
		if (!(uMax > 4 && vMax > 4 && uMax < wG[0] - 5 && vMax < hG[0] - 5)) {
			lastTraceUV = Vec2f(-1, -1);
			lastTracePixelInterval = 0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}
		assert(dist > 0);
	}

	// set OOB if scale change too big.
	if (!(idepth_min < 0 || (ptpMin[2] > 0.75 &&
	                         ptpMin[2] < 1.5))) { //- ptpMin[2] is the depth here. min for idepth, so the depth is max.
		lastTraceUV = Vec2f(-1, -1);
		lastTracePixelInterval = 0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

	// ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
	//- Don't know what a, b means. Very sorry.
	float dx = setting_trace_stepsize * (uMax - uMin);
	float dy = setting_trace_stepsize * (vMax - vMin);

	//- gradH is the patterns gradient covariance sum.
	float a = (Vec2f(dx, dy).transpose() * gradH * Vec2f(dx, dy));
	float b = (Vec2f(dy, -dx).transpose() * gradH * Vec2f(dy, -dx));
	float errorInPixel = 0.2f + 0.2f * (a + b) / a;

	if (errorInPixel * setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max_stereo)) {
		return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
	}

	if (errorInPixel > 10) errorInPixel = 10;

	// ============== do the discrete search ===================
	dx /= dist;
	dy /= dist;

	if (dist > maxPixSearch) {
		uMax = uMin + maxPixSearch * dx;
		vMax = vMin + maxPixSearch * dy;
		dist = maxPixSearch;
	}

	int numSteps = 1.9999f + dist / setting_trace_stepsize;
	Mat22f Rplane = KRKi.topLeftCorner<2, 2>();

	float randShift = uMin * 1000 - floorf(uMin * 1000);
	float ptx = uMin - randShift * dx;
	float pty = vMin - randShift * dy;


	Vec2f rotatePattern[patternNumNCC];
	for (int idx = 0; idx < patternNumNCC; idx++)
		rotatePattern[idx] = Rplane * Vec2f(patternPNCC[idx][0], patternPNCC[idx][1]);

	if (!std::isfinite(dx) || !std::isfinite(dy)) {
		lastTraceUV = Vec2f(-1, -1);
		lastTracePixelInterval = 0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

	float errors[100];
	float bestU = 0, bestV = 0, bestEnergy = 1e10;
	int bestIdx = -1;
	if (numSteps >= 100) numSteps = 99;

	//- Do step searches for the GN optimization initial state.
	for (int i = 0; i < numSteps; i++) {
		float energy = 0;

		Vec15f patternTarget;
		for (int idx = 0; idx < patternNumNCC; idx++) {

			float hitColor = getInterpolatedElement31(frameRight->dI,
			                 (float) (ptx + rotatePattern[idx][0]),
			                 (float) (pty + rotatePattern[idx][1]),
			                 wG[0]);
			patternTarget[idx] = hitColor;
//        if (!std::isfinite(hitColor)) {
//          energy += 1e5;
//          continue;
//        }
//        float residual = hitColor - (float) (aff[0] * color[idx] + aff[1]);
//        float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual); // settings_huberTH == 9
//        energy += hw * residual * residual * (2 - hw);
		}

		patternTarget.normalize();

		energy = 1.0f - patternNCCHostNormalized.dot(patternTarget.transpose());

		errors[i] = energy;
		if (energy < bestEnergy) {
			bestU = ptx;
			bestV = pty;
			bestEnergy = energy;
			bestIdx = i;
		}

		ptx += dx;
		pty += dy;
	}

	// find best score outside a +-2px radius.
	float secondBest = 1e10;
	for (int i = 0; i < numSteps; i++) {
		if ((i < bestIdx - setting_minTraceTestRadius || i > bestIdx + setting_minTraceTestRadius) &&
		    errors[i] < secondBest)
			secondBest = errors[i];
	}
	float newQuality = secondBest / bestEnergy;
	if (newQuality < quality || numSteps > 10) quality = newQuality;

//    std::cout << "bestEnergy: " << bestEnergy << "bestU: " << bestU << std::endl;

	// ============== do GN optimization ===================
	float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0;
	if (setting_trace_GNIterations > 0) bestEnergy = 1e5;
	int gnStepsGood = 0, gnStepsBad = 0;
	for (int it = 0; it < setting_trace_GNIterations; it++) {
		float H = 1, b = 0, energy = 0, J = 0;
		Vec15f patternTarget;
		Vec15f gxTarget;
		Vec15f gyTarget;
		for (int idx = 0; idx < patternNumNCC; idx++) {
			Vec3f hitColor = getInterpolatedElement33(frameRight->dI,
			                 (float) (bestU + rotatePattern[idx][0]),
			                 (float) (bestV + rotatePattern[idx][1]), wG[0]);

//        if (!std::isfinite((float) hitColor[0])) {
//          energy += 1e5;
//          continue;
//        }
//        float residual = hitColor[0] - (aff[0] * color[idx] + aff[1]);
//        float dResdDist = dx * hitColor[1] + dy * hitColor[2];
//        float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
//
//        H += hw * dResdDist * dResdDist;
//        b += hw * residual * dResdDist;
//        energy += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);

			patternTarget[idx] = hitColor[0];
			gxTarget[idx] = hitColor[1];
			gyTarget[idx] = hitColor[2];
		}

		float normTarget = patternTarget.norm();

		energy = 1.0 - patternNCCHostNormalized.dot(patternTarget.transpose()) / normTarget;

		for (int idx = 0; idx < patternNumNCC; idx++) {
			J += -(gxTarget[idx] * dx + gyTarget[idx] * dy)
			     * (1 / normTarget - patternTarget[idx] / (normTarget * normTarget * normTarget))
			     * patternNCCHostNormalized[idx];
		}

		H = J * J;
		b = J * energy;

		if (energy > bestEnergy) {
			gnStepsBad++;
			// do a smaller step from old point.
			stepBack *= 0.5;
			bestU = uBak + stepBack * dx;
			bestV = vBak + stepBack * dy;
		}
		else {
			gnStepsGood++;

			float step = -gnstepsize * b / H;
			if (step < -0.5) step = -0.5;
			else if (step > 0.5) step = 0.5;

			if (!std::isfinite(step)) step = 0;

			uBak = bestU;
			vBak = bestV;
			stepBack = step;

			bestU += step * dx;
			bestV += step * dy;
			bestEnergy = energy;
		}
		//- setting_trace_GNThreshold == 0.1
		if (fabsf(stepBack) < setting_trace_GNThreshold) break;
	}

//    std::cout << "after bestEnergy: " << bestEnergy << "\tbestU: " << bestU << std::endl;

//    if (!(bestEnergy < energyTH * setting_trace_extraSlackOnTH)) { //- This energyTH is constant for all the points.
	if (!(bestEnergy < 0.3)) {
		lastTracePixelInterval = 0;
		lastTraceUV = Vec2f(-1, -1);
		if (lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		else
			return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

	// ============== set new interval ===================
	if (dx * dx > dy * dy) {
		idepth_min_stereo = (pr[2] * (bestU - errorInPixel * dx) - pr[0]) / (Kt[0] - Kt[2] * (bestU - errorInPixel * dx));
		idepth_max_stereo = (pr[2] * (bestU + errorInPixel * dx) - pr[0]) / (Kt[0] - Kt[2] * (bestU + errorInPixel * dx));
	}
	else {
		idepth_min_stereo = (pr[2] * (bestV - errorInPixel * dy) - pr[1]) / (Kt[1] - Kt[2] * (bestV - errorInPixel * dy));
		idepth_max_stereo = (pr[2] * (bestV + errorInPixel * dy) - pr[1]) / (Kt[1] - Kt[2] * (bestV + errorInPixel * dy));
	}
	if (idepth_min_stereo > idepth_max_stereo) std::swap<float>(idepth_min_stereo, idepth_max_stereo);

//  printf("the idpeth_min is %f, the idepth_max is %f \n", idepth_min, idepth_max);

	if (!std::isfinite(idepth_min_stereo) || !std::isfinite(idepth_max_stereo) || (idepth_max_stereo < 0)) {
		lastTracePixelInterval = 0;
		lastTraceUV = Vec2f(-1, -1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

	lastTracePixelInterval = 2 * errorInPixel;
	lastTraceUV = Vec2f(bestU, bestV);
	idepth_stereo = (u_stereo - bestU) / bf;
	return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;
}

#else
/**
 * @brief      { function_description }
 *
 * @param      frame       The frame
 * @param[in]  K           { parameter_description }
 * @param[in]  mode_right  The mode right
 *
 * @return     { description_of_the_return_value }
 * 静态双目匹配，mode_right = true，左图与右图进行匹配，否则为右图与作图匹配,j计算得到idepth_stereo和idepth_stereo_min和idepth_stereo_max
 */
ImmaturePointStatus ImmaturePoint::traceStereo(FrameHessian* frame, Mat33f K, bool mode_right)
{
	// KRKi
	Mat33f KRKi = Mat33f::Identity().cast<float>();
	// Kt
	Vec3f Kt;
	// T between stereo cameras
	Vec3f bl;

	if (mode_right)
	{
		bl << -baseline, 0, 0;
	} else {
		bl << baseline, 0, 0;
	}

	Kt = K * bl;
	// to simplify set aff 1, 0
	Vec2f aff;
	aff << 1, 0;

	// baseline * fx
	float bf = -K(0, 0) * bl[0];

	//重投影
	Vec3f pr = KRKi * Vec3f(u_stereo, v_stereo, 1);
	Vec3f ptpMin = pr + Kt * idepth_min_stereo;

	//投影后的像素坐标
	float uMin = ptpMin[0] / ptpMin[2];
	float vMin = ptpMin[1] / ptpMin[2];

	//投影后的像素坐标是否在图像内,不在则是OOB
	if (!(uMin > 4 && vMin > 4 && uMin < wG[0] - 5 && vMin < hG[0] - 5))
	{
		lastTraceUV = Vec2f(-1, -1);
		lastTracePixelInterval = 0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

	float dist;
	float uMax;
	float vMax;
	Vec3f ptpMax;

	//一开始最大像素搜索图像长宽和的0.027,这为20.088
	float maxPixSearch = (wG[0] + hG[0]) * setting_maxPixSearch;

	//判断该当前idepth_max_stereo是否是NAN,一开始idepth_max_stereo是NAN,不是NAN,则投影,根据最小和最大的像素欧式距离判断是否小于阈值,
	//小于阈值1.5,说明,范围太小IPS_SKIPPED
	if (std::isfinite(idepth_max_stereo))
	{
		//按最大的投影
		ptpMax = pr + Kt * idepth_max_stereo;
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];

		if (!(uMax > 4 && vMax > 4 && uMax < wG[0] - 5 && vMax < hG[0] - 5))
		{
			lastTraceUV = Vec2f(-1, -1);
			lastTracePixelInterval = 0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}

		// ============== check their distance. everything below 2px is OK (-> skip). ===================
		//判断距离
		dist = (uMin - uMax) * (uMin - uMax) + (vMin - vMax) * (vMin - vMax);
		dist = sqrtf(dist);
		if (dist < setting_trace_slackInterval)
		{
//				lastTraceUV_Stereo = Vec2f(uMax+uMin, vMax+vMin)*0.5;
//				lastTracePixelInterval_Stereo=dist;
//				idepth_stereo = (u_stereo - 0.5*(uMax+uMin))/bf;
//				return lastTraceStatus_Stereo = ImmaturePointStatus::IPS_SKIPPED;
			return lastTraceStatus = ImmaturePointStatus ::IPS_SKIPPED;

		}
		assert(dist > 0);
	}
	else
	{
		dist = maxPixSearch;

		// project to arbitrary depth to get direction.
		//一开始最大的设为0.01投影
		ptpMax = pr + Kt * 0.01;
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];

		// direction.
		//最大逆深度和最小逆距离方向
		float dx = uMax - uMin;
		float dy = vMax - vMin;
		//欧式距离的逆
		float d = 1.0f / sqrtf(dx * dx + dy * dy);

		// set to [setting_maxPixSearch].
		//设置新的,以20个像素的步进
		uMax = uMin + dist * dx * d;
		vMax = vMin + dist * dy * d;

		// may still be out!
		//判断是否在图像内
		if (!(uMax > 4 && vMax > 4 && uMax < wG[0] - 5 && vMax < hG[0] - 5))
		{
			lastTraceUV = Vec2f(-1, -1);
			lastTracePixelInterval = 0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}
		assert(dist > 0);
	}

	//set OOB if scale change too big.
	//最小逆深度小于0或者
	if (!(idepth_min < 0 || (ptpMin[2] > 0.75 && ptpMin[2] < 1.5)))
	{
		lastTraceUV = Vec2f(-1, -1);
		lastTracePixelInterval = 0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

	// ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
	float dx = setting_trace_stepsize * (uMax - uMin);
	float dy = setting_trace_stepsize * (vMax - vMin);

	float a = (Vec2f(dx, dy).transpose() * gradH * Vec2f(dx, dy));
	float b = (Vec2f(dy, -dx).transpose() * gradH * Vec2f(dy, -dx));
	float errorInPixel = 0.2f + 0.2f * (a + b) / a;

	if (errorInPixel * setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max_stereo))
	{
//			lastTraceUV_Stereo = Vec2f(uMax+uMin, vMax+vMin)*0.5;
//			lastTracePixelInterval_Stereo=dist;
//			idepth_stereo = (u_stereo - 0.5*(uMax+uMin))/bf;
//			return lastTraceStatus_Stereo = ImmaturePointStatus::IPS_BADCONDITION;
//            lastTraceUV = Vec2f(u, v);
//            lastTracePixelInterval = dist;
		return lastTraceStatus = ImmaturePointStatus ::IPS_BADCONDITION;
	}

	if (errorInPixel > 10) errorInPixel = 10;

	// ============== do the discrete search ===================
	dx /= dist;
	dy /= dist;

	if (dist > maxPixSearch)
	{
		uMax = uMin + maxPixSearch * dx;
		vMax = vMin + maxPixSearch * dy;
		dist = maxPixSearch;
	}

	int numSteps = 1.9999f + dist / setting_trace_stepsize;
	Mat22f Rplane = KRKi.topLeftCorner<2, 2>();

	float randShift = uMin * 1000 - floorf(uMin * 1000);
	float ptx = uMin - randShift * dx;
	float pty = vMin - randShift * dy;

	Vec2f rotatetPattern[MAX_RES_PER_POINT];
	for (int idx = 0; idx < patternNum; idx++)
		rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);

	if (!std::isfinite(dx) || !std::isfinite(dy))
	{
		lastTraceUV = Vec2f(-1, -1);
		lastTracePixelInterval = 0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

	float errors[100];
	float bestU = 0, bestV = 0, bestEnergy = 1e10;
	int bestIdx = -1;
	if (numSteps >= 100) numSteps = 99;

	for (int i = 0; i < numSteps; i++)
	{
		float energy = 0;
		for (int idx = 0; idx < patternNum; idx++)
		{

			float hitColor = getInterpolatedElement31(frame->dI,
			                 (float)(ptx + rotatetPattern[idx][0]),
			                 (float)(pty + rotatetPattern[idx][1]),
			                 wG[0]);

			if (!std::isfinite(hitColor)) {energy += 1e5; continue;}
			float residual = hitColor - (float)(aff[0] * color[idx] + aff[1]);
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw * residual * residual * (2 - hw);
		}

		errors[i] = energy;
		if (energy < bestEnergy)
		{
			bestU = ptx;
			bestV = pty;
			bestEnergy = energy;
			bestIdx = i;
		}

		ptx += dx;
		pty += dy;
	}

	// find best score outside a +-2px radius.
	float secondBest = 1e10;
	for (int i = 0; i < numSteps; i++)
	{
		if ((i < bestIdx - setting_minTraceTestRadius || i > bestIdx + setting_minTraceTestRadius) && errors[i] < secondBest)
			secondBest = errors[i];
	}
	float newQuality = secondBest / bestEnergy;
	if (newQuality < quality || numSteps > 10) quality = newQuality;


	// ============== do GN optimization ===================
	float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0;
	if (setting_trace_GNIterations > 0) bestEnergy = 1e5;
	int gnStepsGood = 0, gnStepsBad = 0;
	for (int it = 0; it < setting_trace_GNIterations; it++)
	{
		float H = 1, b = 0, energy = 0;
		for (int idx = 0; idx < patternNum; idx++)
		{
			Vec3f hitColor = getInterpolatedElement33(frame->dI,
			                 (float)(bestU + rotatetPattern[idx][0]),
			                 (float)(bestV + rotatetPattern[idx][1]), wG[0]);

			if (!std::isfinite((float)hitColor[0])) {energy += 1e5; continue;}
			float residual = hitColor[0] - (aff[0] * color[idx] + aff[1]);
			float dResdDist = dx * hitColor[1] + dy * hitColor[2];
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			H += hw * dResdDist * dResdDist;
			b += hw * residual * dResdDist;
			energy += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
		}

		if (energy > bestEnergy)
		{
			gnStepsBad++;

			// do a smaller step from old point.
			stepBack *= 0.5;
			bestU = uBak + stepBack * dx;
			bestV = vBak + stepBack * dy;
		}
		else
		{
			gnStepsGood++;

			float step = -gnstepsize * b / H;
			if (step < -0.5) step = -0.5;
			else if (step > 0.5) step = 0.5;

			if (!std::isfinite(step)) step = 0;

			uBak = bestU;
			vBak = bestV;
			stepBack = step;

			bestU += step * dx;
			bestV += step * dy;
			bestEnergy = energy;

		}

		if (fabsf(stepBack) < setting_trace_GNThreshold) break;
	}

	if (!(bestEnergy < energyTH * setting_trace_extraSlackOnTH))
	{
		lastTracePixelInterval = 0;
		lastTraceUV = Vec2f(-1, -1);
		if (lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		else
			return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

	// ============== set new interval ===================
	if (dx * dx > dy * dy)
	{
		idepth_min_stereo = (pr[2] * (bestU - errorInPixel * dx) - pr[0]) / (Kt[0] - Kt[2] * (bestU - errorInPixel * dx));
		idepth_max_stereo = (pr[2] * (bestU + errorInPixel * dx) - pr[0]) / (Kt[0] - Kt[2] * (bestU + errorInPixel * dx));
	}
	else
	{
		idepth_min_stereo = (pr[2] * (bestV - errorInPixel * dy) - pr[1]) / (Kt[1] - Kt[2] * (bestV - errorInPixel * dy));
		idepth_max_stereo = (pr[2] * (bestV + errorInPixel * dy) - pr[1]) / (Kt[1] - Kt[2] * (bestV + errorInPixel * dy));
	}
	if (idepth_min_stereo > idepth_max_stereo) std::swap<float>(idepth_min_stereo, idepth_max_stereo);

//  printf("the idpeth_min is %f, the idepth_max is %f \n", idepth_min, idepth_max);

	if (!std::isfinite(idepth_min_stereo) || !std::isfinite(idepth_max_stereo) || (idepth_max_stereo < 0))
	{
		lastTracePixelInterval = 0;
		lastTraceUV = Vec2f(-1, -1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

	lastTracePixelInterval = 2 * errorInPixel;
	lastTraceUV = Vec2f(bestU, bestV);

	//双目得到的逆深度
	idepth_stereo = (u_stereo - bestU) / bf;
	return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;

}
#endif

/*
 * returns
 * * OOB -> point is optimized and marginalized
 * * UPDATED -> point has been updated.
 * * SKIP -> point has not been updated.
 * 更新了lastTraceUV和idepth_min和idepth_max
 */
ImmaturePointStatus ImmaturePoint::traceOn(FrameHessian* frame, Mat33f hostToFrame_KRKi, Vec3f hostToFrame_Kt, Vec2f hostToFrame_affine, CalibHessian* HCalib, bool debugPrint)
{
	//判断点最新的状态
	if (lastTraceStatus == ImmaturePointStatus::IPS_OOB) return lastTraceStatus;

	//是否输出
	debugPrint = false;//rand()%100==0;

	//setting_maxPixSearch=0.027,图像*0.027
	//最大像素搜索范围
	float maxPixSearch = (wG[0] + hG[0]) * setting_maxPixSearch;

	if (debugPrint)
		printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f -> %f. t %f %f %f!\n",
		       u, v,
		       host->shell->id, frame->shell->id,
		       idepth_min, idepth_max,
		       hostToFrame_Kt[0], hostToFrame_Kt[1], hostToFrame_Kt[2]);

//	const float stepsize = 1.0;				// stepsize for initial discrete search.
//	const int GNIterations = 3;				// max # GN iterations
//	const float GNThreshold = 0.1;				// GN stop after this stepsize.
//	const float extraSlackOnTH = 1.2;			// for energy-based outlier check, be slightly more relaxed by this factor.
//	const float slackInterval = 0.8;			// if pixel-interval is smaller than this, leave it be.
//	const float minImprovementFactor = 2;		// if pixel-interval is smaller than this, leave it be.

	// ============== project min and max. return if one of them is OOB ===================
	//重投影根据最小和最大逆深度
	//重投影后点位姿
	Vec3f pr = hostToFrame_KRKi * Vec3f(u, v, 1);
	//最小逆深度位姿
	Vec3f ptpMin = pr + hostToFrame_Kt * idepth_min;

	//最小逆深度的像素位姿
	float uMin = ptpMin[0] / ptpMin[2];
	float vMin = ptpMin[1] / ptpMin[2];

	//    float idepth_min_project = 1.0f / ptpMin[2];
	//    printf("the idepth min project %f \n", idepth_min_project);
	//限制像素位姿，超出图像
	if (!(uMin > 4 && vMin > 4 && uMin < wG[0] - 5 && vMin < hG[0] - 5))
	{
		if (debugPrint) printf("OOB uMin %f %f - %f %f %f (id %f-%f)!\n",
			                       u, v, uMin, vMin,  ptpMin[2], idepth_min, idepth_max);
		lastTraceUV = Vec2f(-1, -1);
		lastTracePixelInterval = 0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

	//最大和最小逆深度像素坐标欧式距离
	float dist;
	float uMax;
	float vMax;
	Vec3f ptpMax;
	//判断最大逆深度值
	if (std::isfinite(idepth_max))
	{
		//最大逆深度位姿
		ptpMax = pr + hostToFrame_Kt * idepth_max;
		//最大逆深度像素坐标
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];

		//判断坐标，超出图像
		if (!(uMax > 4 && vMax > 4 && uMax < wG[0] - 5 && vMax < hG[0] - 5))
		{
			if (debugPrint) printf("OOB uMax  %f %f - %f %f!\n", u, v, uMax, vMax);
			lastTraceUV = Vec2f(-1, -1);
			lastTracePixelInterval = 0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}

		// ============== check their distance. everything below 2px is OK (-> skip). ===================
		// 判断这两个点的像素坐标距离
		dist = (uMin - uMax) * (uMin - uMax) + (vMin - vMax) * (vMin - vMax);
		dist = sqrtf(dist);

		//距离小于阈值，这里设为1.5，过小
		if (dist < setting_trace_slackInterval)
		{
			if (debugPrint)
				printf("TOO CERTAIN ALREADY (dist %f)!\n", dist);

			//则直接设为平均值，是IPS_SKIPPED，深度范围过小
			lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;
			lastTracePixelInterval = dist;
			return lastTraceStatus = ImmaturePointStatus::IPS_SKIPPED;
		}
		assert(dist > 0);
	}
	else
	{
		//最大像素搜索距离
		dist = maxPixSearch;

		// project to arbitrary depth to get direction.
		// 重投影该点，最大逆深度设为0.01
		ptpMax = pr + hostToFrame_Kt * 0.01;
		uMax = ptpMax[0] / ptpMax[2];
		vMax = ptpMax[1] / ptpMax[2];

		// direction.
		// 距离
		float dx = uMax - uMin;
		float dy = vMax - vMin;
		float d = 1.0f / sqrtf(dx * dx + dy * dy);

		// set to [setting_maxPixSearch].
		// 根据最大距离，设置最大距离
		uMax = uMin + dist * dx * d;
		vMax = vMin + dist * dy * d;

		// may still be out!	判断是否超出图像
		if (!(uMax > 4 && vMax > 4 && uMax < wG[0] - 5 && vMax < hG[0] - 5))
		{
			if (debugPrint) printf("OOB uMax-coarse %f %f %f!\n", uMax, vMax,  ptpMax[2]);
			lastTraceUV = Vec2f(-1, -1);
			lastTracePixelInterval = 0;
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		}
		assert(dist > 0);
	}

	// set OOB if scale change too big.
	// 若最小逆深度小于0，或者最小逆深度算出来的深度在0.75-1.5之内，说明深度在0.666-1.333米内都是有问题的
	// 则也是超出范围
	if (!(idepth_min < 0 || (ptpMin[2] > 0.75 && ptpMin[2] < 1.5)))
	{
		if (debugPrint) printf("OOB SCALE %f %f %f!\n", uMax, vMax,  ptpMin[2]);
		lastTraceUV = Vec2f(-1, -1);
		lastTracePixelInterval = 0;
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}


	// ============== compute error-bounds on result in pixel. if the new interval is not at least 1/2 of the old, SKIP ===================
	//计算像素结果的误差范围。 如果新的时间间隔不是旧的SKIP的1/2
	//setting_trace_stepsize=1.0
	float dx = setting_trace_stepsize * (uMax - uMin);
	float dy = setting_trace_stepsize * (vMax - vMin);

	float a = (Vec2f(dx, dy).transpose() * gradH * Vec2f(dx, dy));
	float b = (Vec2f(dy, -dx).transpose() * gradH * Vec2f(dy, -dx));
	float errorInPixel = 0.2f + 0.2f * (a + b) / a;

	//像素误差*setting_trace_minImprovementFactor(2) >，则这点也是bad
	if (errorInPixel * setting_trace_minImprovementFactor > dist && std::isfinite(idepth_max))
	{
		if (debugPrint)
			printf("NO SIGNIFICANT IMPROVMENT (%f)!\n", errorInPixel);
		lastTraceUV = Vec2f(uMax + uMin, vMax + vMin) * 0.5;
		lastTracePixelInterval = dist;
		return lastTraceStatus = ImmaturePointStatus::IPS_BADCONDITION;
	}

	if (errorInPixel > 10) errorInPixel = 10;

	// ============== do the discrete search ===================
	// 这里根据离散的搜索搜索出先验最佳像素坐标
	//搜索步长，开始搜索，在这里和后面高斯牛顿迭代中用于步长
	dx /= dist;
	dy /= dist;

	//距离大于最大搜索距离，即图像*0.0027
	if (debugPrint)
		printf("trace pt (%.1f %.1f) from frame %d to %d. Range %f (%.1f %.1f) -> %f (%.1f %.1f)! ErrorInPixel %.1f!\n",
		       u, v,
		       host->shell->id, frame->shell->id,
		       idepth_min, uMin, vMin,
		       idepth_max, uMax, vMax,
		       errorInPixel
		      );

	if (dist > maxPixSearch)
	{
		uMax = uMin + maxPixSearch * dx;
		vMax = vMin + maxPixSearch * dy;
		dist = maxPixSearch;
	}
	//1.999+距离，迭代次数
	int numSteps = 1.9999f + dist / setting_trace_stepsize;
	//旋转平面，前两个????
	Mat22f Rplane = hostToFrame_KRKi.topLeftCorner<2, 2>();

	//插个值后的像素坐标，从最远开始搜索
	float randShift = uMin * 1000 - floorf(uMin * 1000);
	float ptx = uMin - randShift * dx;
	float pty = vMin - randShift * dy;

	//旋转模式
	Vec2f rotatetPattern[MAX_RES_PER_POINT];
	//该模式xia每个点根据旋转进行变换后的点
	for (int idx = 0; idx < patternNum; idx++)
		rotatetPattern[idx] = Rplane * Vec2f(patternP[idx][0], patternP[idx][1]);

	if (!std::isfinite(dx) || !std::isfinite(dy))
	{
		//printf("COUGHT INF / NAN dxdy (%f %f)!\n", dx, dx);

		lastTracePixelInterval = 0;
		lastTraceUV = Vec2f(-1, -1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
	}

	//每次迭代中的误差
	float errors[100];
	//寻找最小的误差能量
	float bestU = 0, bestV = 0, bestEnergy = 1e10;
	int bestIdx = -1;
	//限制最大步骤
	if (numSteps >= 100) numSteps = 99;

	for (int i = 0; i < numSteps; i++)
	{
		float energy = 0;
		//遍历模式下每个点
		for (int idx = 0; idx < patternNum; idx++)
		{
			//灰度值
			float hitColor = getInterpolatedElement31(frame->dI,
			                 (float)(ptx + rotatetPattern[idx][0]),
			                 (float)(pty + rotatetPattern[idx][1]),
			                 wG[0]);
			//无灰度值，则加上一个很大的值
			if (!std::isfinite(hitColor)) {energy += 1e5; continue;}
			//灰度残差
			float residual = hitColor - (float)(hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
			//huber函数下的energy,加上能量
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			energy += hw * residual * residual * (2 - hw);
		}

		if (debugPrint)
			printf("step %.1f %.1f (id %f): energy = %f!\n",
			       ptx, pty, 0.0f, energy);

		//找到最小的误差
		errors[i] = energy;
		if (energy < bestEnergy)
		{
			bestU = ptx;
			bestV = pty;
			bestEnergy = energy;
			bestIdx = i;
		}
		//步长
		ptx += dx;
		pty += dy;
	}


	// find best score outside a +-2px radius.
	float secondBest = 1e10;
	// 第二次搜索，第二次搜索最小误差
	for (int i = 0; i < numSteps; i++)
	{
		//在第一次搜索的最佳id周围的半径setting_minTraceTestRadius(2)之间，误差在1e10之间
		if ((i < bestIdx - setting_minTraceTestRadius || i > bestIdx + setting_minTraceTestRadius) && errors[i] < secondBest)
			secondBest = errors[i];
	}
	//这里的意思是，若bestEnergy>1e10,则1e10/bestEnergy
	float newQuality = secondBest / bestEnergy;

	//更新最新的质量
	if (newQuality < quality || numSteps > 10) quality = newQuality;


	// ============== do GN optimization ===================
	// 进行高斯牛顿迭代
	// 迭代之前的最佳像素坐标
	float uBak = bestU, vBak = bestV, gnstepsize = 1, stepBack = 0;
	//高斯迭代次数3，这里最佳误差设为1e5
	if (setting_trace_GNIterations > 0) bestEnergy = 1e5;
	//好的迭代，换的迭代
	int gnStepsGood = 0, gnStepsBad = 0;
	//迭代
	for (int it = 0; it < setting_trace_GNIterations; it++)
	{
		float H = 1, b = 0, energy = 0;

		//模式的每个点
		for (int idx = 0; idx < patternNum; idx++)
		{
			Vec3f hitColor = getInterpolatedElement33(frame->dI,
			                 (float)(bestU + rotatetPattern[idx][0]),
			                 (float)(bestV + rotatetPattern[idx][1]), wG[0]);

			if (!std::isfinite((float)hitColor[0])) {energy += 1e5; continue;}
			//残差
			float residual = hitColor[0] - (hostToFrame_affine[0] * color[idx] + hostToFrame_affine[1]);
			//梯度导数
			float dResdDist = dx * hitColor[1] + dy * hitColor[2];
			//huber
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

			//Hessian矩阵
			H += hw * dResdDist * dResdDist;
			//残差导数
			b += hw * residual * dResdDist;

			//误差能量值+= w^2*hw*r^2*(2-hw)
			energy += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
		}

		//若新算出来的误差大于最小误差
		if (energy > bestEnergy)
		{
			//不好的迭代次数++
			gnStepsBad++;

			// do a smaller step from old point.
			// 补长减小一半
			stepBack *= 0.5;
			//更新像素坐标
			bestU = uBak + stepBack * dx;
			bestV = vBak + stepBack * dy;
			if (debugPrint)
				printf("GN BACK %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
				       it, energy, H, b, stepBack,
				       uBak, vBak, bestU, bestV);
		}
		else
		{
			//好的迭代次数++
			gnStepsGood++;

			//补偿=b/H=residual/dResdDist
			float step = -gnstepsize * b / H;
			//限制范围
			if (step < -0.5) step = -0.5;
			else if (step > 0.5) step = 0.5;

			if (!std::isfinite(step)) step = 0;

			uBak = bestU;
			vBak = bestV;

			//更新步长
			stepBack = step;

			//更新最佳像素坐标和最佳误差能量值
			bestU += step * dx;
			bestV += step * dy;
			bestEnergy = energy;

			if (debugPrint)
				printf("GN step %d: E %f, H %f, b %f. id-step %f. UV %f %f -> %f %f.\n",
				       it, energy, H, b, step,
				       uBak, vBak, bestU, bestV);
		}

		//当迭代步长小于阈值，跳出
		if (fabsf(stepBack) < setting_trace_GNThreshold) break;
	}


	// ============== detect energy-based outlier. ===================
//	float absGrad0 = getInterpolatedElement(frame->absSquaredGrad[0],bestU, bestV, wG[0]);
//	float absGrad1 = getInterpolatedElement(frame->absSquaredGrad[1],bestU*0.5-0.25, bestV*0.5-0.25, wG[1]);
//	float absGrad2 = getInterpolatedElement(frame->absSquaredGrad[2],bestU*0.25-0.375, bestV*0.25-0.375, wG[2]);
//	基于能量检测outlier
//	最小误差能量大于阈值，这里8*
	if (!(bestEnergy < energyTH * setting_trace_extraSlackOnTH))
//			|| (absGrad0*areaGradientSlackFactor < host->frameGradTH
//		     && absGrad1*areaGradientSlackFactor < host->frameGradTH*0.75f
//			 && absGrad2*areaGradientSlackFactor < host->frameGradTH*0.50f))
	{
		if (debugPrint)
			printf("OUTLIER!\n");

		//则是IPS_OOB或者IPS_OUTLIER
		lastTracePixelInterval = 0;
		lastTraceUV = Vec2f(-1, -1);
		//上一时刻为IPS_OUTLIER，则IPS_OOB
		if (lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER)
			return lastTraceStatus = ImmaturePointStatus::IPS_OOB;
		else
			return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}


	// ============== set new interval ===================
	// 设置新的间隔，即设置最小和最大的逆深度
	// x方向梯度较大
	if (dx * dx > dy * dy)
	{
		//则根据x方向设置最新的最小逆深度和最大逆深度
		idepth_min = (pr[2] * (bestU - errorInPixel * dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU - errorInPixel * dx));
		idepth_max = (pr[2] * (bestU + errorInPixel * dx) - pr[0]) / (hostToFrame_Kt[0] - hostToFrame_Kt[2] * (bestU + errorInPixel * dx));
	}
	else
	{
		//则根据x方向设置最新的最小逆深度和最大逆深度
		idepth_min = (pr[2] * (bestV - errorInPixel * dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV - errorInPixel * dy));
		idepth_max = (pr[2] * (bestV + errorInPixel * dy) - pr[1]) / (hostToFrame_Kt[1] - hostToFrame_Kt[2] * (bestV + errorInPixel * dy));
	}
	if (idepth_min > idepth_max) std::swap<float>(idepth_min, idepth_max);

	//判断最新和最大的逆深度
	if (!std::isfinite(idepth_min) || !std::isfinite(idepth_max) || (idepth_max < 0))
	{
		//printf("COUGHT INF / NAN minmax depth (%f %f)!\n", idepth_min, idepth_max);

		lastTracePixelInterval = 0;
		lastTraceUV = Vec2f(-1, -1);
		return lastTraceStatus = ImmaturePointStatus::IPS_OUTLIER;
	}

	//最新的逆深度间隔
	lastTracePixelInterval = 2 * errorInPixel;
	//最佳的重投影的像素坐标
	lastTraceUV = Vec2f(bestU, bestV);

	//运行到这一步说明该点好的
	return lastTraceStatus = ImmaturePointStatus::IPS_GOOD;

}

/**
 * [ImmaturePoint::getdPixdd description]
 * @param  HCalib [description]
 * @param  tmpRes [description]
 * @param  idepth [description]
 * @return        [description]
 * 获取像素梯度
 */
float ImmaturePoint::getdPixdd(
  CalibHessian *  HCalib,
  ImmaturePointTemporaryResidual* tmpRes,
  float idepth)
{
	//目标帧的预计值
	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);
	//预计的t
	const Vec3f &PRE_tTll = precalc->PRE_tTll;
	float drescale, u = 0, v = 0, new_idepth;
	float Ku, Kv;
	Vec3f KliP;

	//重投影点
	projectPoint(this->u, this->v, idepth, 0, 0, HCalib,
	             precalc->PRE_RTll, PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth);

	//重投影后的点的图像几何梯度
	float dxdd = (PRE_tTll[0] - PRE_tTll[2] * u) * HCalib->fxl();
	float dydd = (PRE_tTll[1] - PRE_tTll[2] * v) * HCalib->fyl();

	//计算后的逆深度*梯度
	return drescale * sqrtf(dxdd * dxdd + dydd * dydd);
}

/**
 * [ImmaturePoint::calcResidual description]
 * @param  HCalib         [description]
 * @param  outlierTHSlack [description]
 * @param  tmpRes         [description]
 * @param  idepth         [description]
 * @return                [description]
 *
 * 计算残差
 */
float ImmaturePoint::calcResidual(
  CalibHessian *  HCalib, const float outlierTHSlack,
  ImmaturePointTemporaryResidual* tmpRes,
  float idepth)
{
	//
	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

	float energyLeft = 0;
	//
	const Eigen::Vector3f* dIl = tmpRes->target->dI;
	const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
	const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
	Vec2f affLL = precalc->PRE_aff_mode;

	//残差
	for (int idx = 0; idx < patternNum; idx++)
	{
		float Ku, Kv;
		if (!projectPoint(this->u + patternP[idx][0], this->v + patternP[idx][1], idepth, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
		{return 1e10;}

		Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
		if (!std::isfinite((float)hitColor[0])) {return 1e10;}
		//if(benchmarkSpecialOption==5) hitColor = (getInterpolatedElement13BiCub(tmpRes->target->I, Ku, Kv, wG[0]));

		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		energyLeft += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);
	}

	if (energyLeft > energyTH * outlierTHSlack)
	{
		energyLeft = energyTH * outlierTHSlack;
	}
	return energyLeft;
}

/**
 * [ImmaturePoint::linearizeResidual description]
 * @param  HCalib         [description]
 * @param  outlierTHSlack [description]
 * @param  tmpRes         [description]
 * @param  Hdd            [description]
 * @param  bd             [description]
 * @param  idepth         [description]
 * @return                [description]
 * 线性化点残差
 */
double ImmaturePoint::linearizeResidual(
  CalibHessian *  HCalib, const float outlierTHSlack,
  ImmaturePointTemporaryResidual* tmpRes,
  float &Hdd, float &bd,
  float idepth)
{
	if (tmpRes->state_state == ResState::OOB)
	{ tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy; }

	FrameFramePrecalc* precalc = &(host->targetPrecalc[tmpRes->target->idx]);

	// check OOB due to scale angle change.

	float energyLeft = 0;
	//目标帧的灰度值和梯度值
	const Eigen::Vector3f* dIl = tmpRes->target->dI;
	//主导帧和目标帧的位姿变换
	const Mat33f &PRE_RTll = precalc->PRE_RTll;
	const Vec3f &PRE_tTll = precalc->PRE_tTll;
	//const float * const Il = tmpRes->target->I;

	//光度变换
	Vec2f affLL = precalc->PRE_aff_mode;

	//模式
	for (int idx = 0; idx < patternNum; idx++)
	{
		int dx = patternP[idx][0];
		int dy = patternP[idx][1];

		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		//重投影点
		if (!projectPoint(this->u, this->v, idepth, dx, dy, HCalib,
		                  PRE_RTll, PRE_tTll, drescale, u, v, Ku, Kv, KliP, new_idepth))
		{tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}

		//获取灰度值
		Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));

		if (!std::isfinite((float)hitColor[0])) {tmpRes->state_NewState = ResState::OOB; return tmpRes->state_energy;}
		//残差
		float residual = hitColor[0] - (affLL[0] * color[idx] + affLL[1]);

		//huber
		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		//总误差
		energyLeft += weights[idx] * weights[idx] * hw * residual * residual * (2 - hw);

		// depth derivatives.
		//逆深度的导数
		float dxInterp = hitColor[1] * HCalib->fxl();
		float dyInterp = hitColor[2] * HCalib->fyl();
		float d_idepth = derive_idepth(PRE_tTll, u, v, dx, dy, dxInterp, dyInterp, drescale);

		hw *= weights[idx] * weights[idx];

		//H和b
		Hdd += (hw * d_idepth) * d_idepth;
		bd += (hw * residual) * d_idepth;
	}

	//误差是否大于阈值，则位outlier
	if (energyLeft > energyTH * outlierTHSlack)
	{
		energyLeft = energyTH * outlierTHSlack;
		tmpRes->state_NewState = ResState::OUTLIER;
	}
	else
	{
		tmpRes->state_NewState = ResState::IN;
	}

	//设置误差
	tmpRes->state_NewEnergy = energyLeft;
	return energyLeft;
}

// void ImmaturePoint::ComputeWorldPos()
// {
//   if (!host)
//     return;
//   SE3 Twc = host->shell->camToWorldOpti;
//   Vec3 Kip = 1.0 / this->idepth * Vec3(
//                fxiG[0]  * this->u + cxiG[0],
//              fyiG[0]  * this->v + cyiG[0],
//              1);
//   mWorldPos = Twc * Kip;
// }

bool ImmaturePoint::ComputePos(Vec3& pose)
{
	if (!std::isfinite(idepth_stereo))
		return false;

	pose = 1.0 / this->idepth_stereo * Vec3(
	         fxiG[0]  * this->u + cxiG[0],
	         fyiG[0]  * this->v + cyiG[0],
	         1);

	return true;
}

}
