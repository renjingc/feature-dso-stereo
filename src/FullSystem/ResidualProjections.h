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


#pragma once

#include "util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "util/settings.h"

namespace fdso
{

/**
 * [derive_idepth description]
 * @param  t        [description]
 * @param  u        [description]
 * @param  v        [description]
 * @param  dx       [description]
 * @param  dy       [description]
 * @param  dxInterp [description]
 * @param  dyInterp [description]
 * @param  drescale [description]
 * @return          [description]
 */
EIGEN_STRONG_INLINE float derive_idepth(
		const Vec3f &t, const float &u, const float &v,
		const int &dx, const int &dy, const float &dxInterp,
		const float &dyInterp, const float &drescale)
{
	return (dxInterp*drescale * (t[0]-t[2]*u)
			+ dyInterp*drescale * (t[1]-t[2]*v))*SCALE_IDEPTH;
}


/**
 * [projectPoint description]
 * @param  u_pt   [description]
 * @param  v_pt   [description]
 * @param  idepth [description]
 * @param  KRKi   [description]
 * @param  Kt     [description]
 * @param  Ku     [description]
 * @param  Kv     [description]
 * @return        [description]
 */
EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt,const float &v_pt,
		const float &idepth,
		const Mat33f &KRKi, const Vec3f &Kt,
		float &Ku, float &Kv)
{
	//像素坐标投影到当前坐标系下3D点
	Vec3f ptp = KRKi * Vec3f(u_pt,v_pt, 1) + Kt*idepth;
	//在投影到相机坐标系的Ku,Kv,即未除去内参的
	Ku = ptp[0] / ptp[2];
	Kv = ptp[1] / ptp[2];

	//判断这个点是否在合理范围内
	return Ku>1.1f && Kv>1.1f && Ku<wM3G && Kv<hM3G;
}

/**
 * [projectPoint description]
 * @param  u_pt       [description]
 * @param  v_pt       [description]
 * @param  idepth     [description]
 * @param  dx         [description]
 * @param  dy         [description]
 * @param  HCalib     [description]
 * @param  R          [description]
 * @param  t          [description]
 * @param  drescale   [description]
 * @param  u          [description]
 * @param  v          [description]
 * @param  Ku         [description]
 * @param  Kv         [description]
 * @param  KliP       [description]
 * @param  new_idepth [description]
 * @return            [description]
 */
EIGEN_STRONG_INLINE bool projectPoint(
		const float &u_pt,const float &v_pt,
		const float &idepth,
		const int &dx, const int &dy,
		CalibHessian* const &HCalib,
		const Mat33f &R, const Vec3f &t,
		float &drescale, float &u, float &v,
		float &Ku, float &Kv, Vec3f &KliP, float &new_idepth)
{
	//矫正后的相机坐标系的点的位姿
	KliP = Vec3f(
			(u_pt+dx-HCalib->cxl())*HCalib->fxli(),
			(v_pt+dy-HCalib->cyl())*HCalib->fyli(),
			1);
	//变换
	Vec3f ptp = R * KliP + t*idepth;
	//逆深度
	drescale = 1.0f/ptp[2];
	//新的逆深度
	new_idepth = idepth*drescale;

	//判断逆深度是否大于０
	if(!(drescale>0)) return false;

	//变换后的坐标
	u = ptp[0] * drescale;
	v = ptp[1] * drescale;

	//内参矫正
	Ku = u*HCalib->fxl() + HCalib->cxl();
	Kv = v*HCalib->fyl() + HCalib->cyl();

	return Ku>1.1f && Kv>1.1f && Ku<wM3G && Kv<hM3G;
}

}

