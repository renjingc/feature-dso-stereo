#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/FrameHessian.h"
#include "FullSystem/PointHessian.h"
#include "FullSystem/CalibHessian.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ImmaturePoint.h"
#include "util/nanoflann.h"


namespace fdso
{

/**
 * 初始化，a和b为0，初始后面帧相对于第一帧参考帧的位姿变换
 */
CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0, 0), thisToNext(SE3())
{
	//初始化每层的点
	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
	{
		int wl = ww >> lvl;
		int hl = hh >> lvl;
		points[lvl] = 0;
		numPoints[lvl] = 0;
		idepth[lvl] = new float[wl * hl];
	}

	//初始雅克比,一个与原图相同大小的10向量
	JbBuffer = new Vec10f[ww * hh];
	JbBuffer_new = new Vec10f[ww * hh];

	//初始化当前帧ID号
	frameID = -1;
	//是否在优化的时候fix住Affine，不优化a和b
	fixAffine = true;
	//不输出Debug
	printDebug = false;

	//8*8的对角矩阵
	/*
	*[1.0
	*       1.0
	*              1.0
	*                      0.5
	*                              0.5
	*                                      0.5
	*                                              10
	*                                                      1000]
	*/
	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
}

/**
 * 释放
 */
CoarseInitializer::~CoarseInitializer()
{
	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
	{
		if (points[lvl] != 0) delete[] points[lvl];
		delete[] idepth[lvl];
	}

	delete[] JbBuffer;
	delete[] JbBuffer_new;
}

/**
 * [CoarseInitializer::trackFrame description]
 * @param  newFrameHessian       [description]
 * @param  newFrameHessian_Right [description]
 * @param  wraps                 [description]
 * @return                       [description]
 */
bool CoarseInitializer::trackFrame(FrameHessian* newFrameHessian, FrameHessian* newFrameHessian_Right, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
	//新一帧的Hessian
	newFrame = newFrameHessian;

	//是否显示
	for (IOWrap::Output3DWrapper* ow : wraps) {
		//将新一帧插入显示类中
		//ow->pushLiveFrame(newFrameHessian);
		ow->pushStereoLiveFrame(newFrameHessian, newFrameHessian_Right);
	}

	//最大迭代次数
	int maxIterations[] = {5, 5, 10, 30, 50};

	//每层的log能量
	alphaK = 2.5 * 2.5; //*freeDebugParam1*freeDebugParam1;
	alphaW = 150 * 150; //*freeDebugParam2*freeDebugParam2;
	//当前点的深度为当前这个点的深度与中值点深度互补,时中值点深度权重
	regWeight = 0.8;//*freeDebugParam4;
	couplingWeight = 1;//*freeDebugParam5;

	//未有第一次平移够大的时候，则平移量都先设为0
	//点的逆深度都先设为1，Hessian矩阵也都先设为0
	if (!snapped)
	{
		//变换矩阵的平移设置为0
		thisToNext.translation().setZero();
		for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
		{
			//得到每一层的点数
			int npts = numPoints[lvl];
			//得到每一层的点
			Pnt* ptsl = points[lvl];
			for (int i = 0; i < npts; i++)
			{
				//每一层的点初始深度iR,idepth_new,Hessian值
				ptsl[i].iR = 1;
				ptsl[i].idepth_new = 1;
				ptsl[i].lastHessian = 0;
			}
		}
	}

	//参考帧到当前帧的值，开始为０
	SE3 refToNew_current = thisToNext;
	//参考帧到当前帧的Affine
	AffLight refToNew_aff_current = thisToNext_aff;

	//如果两帧的曝光时间都大于０,则approximation＝log(新的一帧/第一帧)
	if (firstFrame->ab_exposure > 0 && newFrame->ab_exposure > 0)
		refToNew_aff_current = AffLight(logf(newFrame->ab_exposure /  firstFrame->ab_exposure), 0); // coarse approximation.

	//残差设为零
	Vec3f latestRes = Vec3f::Zero();
	//每一层都去迭代优化初始化
	//从最上一层往下
	for (int lvl = pyrLevelsUsed - 1; lvl >= 0; lvl--)
	{
		//如果这层不是最高层，则要根据上一层传递
		//先根据上一层的父点的深度iR和当前层点的深度计算当前层点的深度，iR,iDepth,iDepth_new
		//再更新当前层的点的深度iR,再将这个点深度idepth与周围10个点iR的中值深度点进行互补滤波,设置iR
		if (lvl < pyrLevelsUsed - 1)
			propagateDown(lvl + 1);

		//Hessian矩阵和B矩阵
		Mat88f H, Hsc; Vec8f b, bsc;
		//重置每个点的iR,iDepth,iDepth_new，根据周围的10个点的iR取平均
		resetPoints(lvl);
		//计算残差，当前层的，H,b,Hsc,bsc,当前帧相对与参考帧的位姿变换和光度参数
		//初始残差,返回总误差能量值，返回alphaEnergy(权重*平移量*点数)，总的点数
		Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
		//下一步，使用点的new更新无new的
		//更新pts[i].energy,pts[i].isGood,pts[i].idepth,pts[i].lastHessian,JbBuffer
		applyStep(lvl);

		//LM算法中控制下降速度
		//如果下降太快，使用较小的λ，使之更接近高斯牛顿法
		//如果下降太慢，使用较大的λ，使之更接近梯度下降法
		//这里迭代成功，则使用较大的λ，失败，则使用较小的λ
		float lambda = 0.1;
		float eps = 1e-4;
		int fails = 0;

		if (printDebug)
		{
			printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
			       lvl, 0, lambda,
			       "INITIA",
			       //总的误差能量值/总的点数
			       sqrtf((float)(resOld[0] / resOld[2])),
			       //平移量
			       sqrtf((float)(resOld[1] / resOld[2])),
			       sqrtf((float)(resOld[0] / resOld[2])),
			       sqrtf((float)(resOld[1] / resOld[2])),
			       (resOld[0] + resOld[1]) / resOld[2],
			       (resOld[0] + resOld[1]) / resOld[2],
			       0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() << "\n";
		}
		//迭代次数
		int iteration = 0;
		while (true)
		{
			/**
			 * 8*8矩阵
			 */
			Mat88f Hl = H;
			//对角线*(1+lambda)
			for (int i = 0; i < 8; i++)
				Hl(i, i) *= (1 + lambda);
			//HI=HI-Hsc*1/(1+lambda)
			//梯度下降，lambda越大则更新越小
			Hl -= Hsc * (1 / (1 + lambda));

			//bI=b-bsc*1/(1+lambda)
			Vec8f bl = b - bsc * (1 / (1 + lambda));

			//乘上权重
			Hl = wM * Hl * wM * (0.01f / (w[lvl] * h[lvl]));
			bl = wM * bl * (0.01f / (w[lvl] * h[lvl]));

			//8个变量
			Vec8f inc;
			//是否fix住光度GAffine，不进行a和b的优化
			//计算H inc =-b
			//inc=-H^-1 * b
			if (fixAffine)
			{
				inc.head<6>() = - (wM.toDenseMatrix().topLeftCorner<6, 6>() * (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
				inc.tail<2>().setZero();
			}
			else
				inc = - (wM * (Hl.ldlt().solve(bl)));	//=-H^-1 * b.

			//先指数映射,前6个变量为位姿的左扰动，更新位姿
			SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
			//新的a和b
			AffLight refToNew_aff_new = refToNew_aff_current;
			//更新a和b
			refToNew_aff_new.a += inc[6];
			refToNew_aff_new.b += inc[7];

			//更新每个点的idepth_new
			doStep(lvl, lambda, inc);

			//新的H,b矩阵
			Mat88f H_new, Hsc_new; Vec8f b_new, bsc_new;
			//再计算一次残差，返回总误差能量值，返回alphaEnergy(权重*平移量*点数)，总的点数
			Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
			//计算当前层总的深度值能量函数
			Vec3f regEnergy = calcEC(lvl);

			//新的总误差能量值，新的alphaEnergy(权重*平移量*点数)值，新的深度和
			float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
			//旧的残差和
			float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);

			//残差是否减小
			bool accept = eTotalOld > eTotalNew;

			if (printDebug)
			{
				printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
				       lvl, iteration, lambda,
				       (accept ? "ACCEPT" : "REJECT"),
				       sqrtf((float)(resOld[0] / resOld[2])),
				       sqrtf((float)(regEnergy[0] / regEnergy[2])),
				       sqrtf((float)(resOld[1] / resOld[2])),
				       sqrtf((float)(resNew[0] / resNew[2])),
				       sqrtf((float)(regEnergy[1] / regEnergy[2])),
				       sqrtf((float)(resNew[1] / resNew[2])),
				       eTotalOld / resNew[2],
				       eTotalNew / resNew[2],
				       inc.norm());
				std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() << "\n";
			}

			//如果残差减小
			if (accept)
			{
				//alphaEnergy=这一层点数*6.25，即残差一定大，则snapped=true
				//说明平移够大了，则可以进行初始化了
				if (resNew[1] == alphaK * numPoints[lvl])
					snapped = true;

				//更新几个Hessian,b矩阵和残差，和位姿变换
				H = H_new;
				b = b_new;
				Hsc = Hsc_new;
				bsc = bsc_new;
				resOld = resNew;
				refToNew_aff_current = refToNew_aff_new;
				refToNew_current = refToNew_new;
				//更新下一步，使用点的new更新无new的
				//更新pts[i].energy,pts[i].isGood,pts[i].idepth,pts[i].lastHessian,JbBuffer
				applyStep(lvl);
				//更新当前层的点的深度iR,再将这个点深度idepth与周围10个点iR的中值深度点进行互补滤波,设置iR
				optReg(lvl);
				//lambda缩小一倍
				lambda *= 0.5;
				//设为成功
				fails = 0;
				//lambda最小为0.0001,14次成功迭代后
				if (lambda < 0.0001)
					lambda = 0.0001;
			}
			else
			{
				//失败次数++
				fails++;
				//加大lambda
				lambda *= 4;
				if (lambda > 10000)
					lambda = 10000;
			}

			//是否结束迭代
			bool quitOpt = false;

			//如果档次迭代归一化后结果小与一定阈值，或迭代次数大于了当曾设置的迭代次数
			//或者失败次数大于了2次后，退出迭代
			if (!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2)
			{
				Mat88f H, Hsc; Vec8f b, bsc;
				quitOpt = true;
			}

			if (quitOpt) break;
			//迭代次数++
			iteration++;
		}
		//更新最新的残差
		latestRes = resOld;
	}
	//下一帧相对于参考帧的位姿变换
	thisToNext = refToNew_current;
	thisToNext_aff = refToNew_aff_current;

	//向上传递
	for (int i = 0; i < pyrLevelsUsed - 1; i++)
		propagateUp(i);

	//帧++
	frameID++;
//    printf("frameID is %d \n", frameID);
	//若snapped=false,snappedAt=0
	//在snapped=false之前，snappedAd都等于0
	if (!snapped) snappedAt = 0;

	//snappedAt=true并且之前有过snappedAt=false,则snappedAt=当前帧
	///第一次平移够了，则snappedAt为当前帧的ID
	if (snapped && snappedAt == 0)
		snappedAt = frameID;

	debugPlot(0, wraps);

	//当前snapped=true，即有过一次平移够大了，且当前帧比那次平移够大的帧之间有了5帧了
	return snapped && frameID > snappedAt + 1;
}

/**
 * [CoarseInitializer::debugPlot description]
 * @param lvl   [description]
 * @param wraps [description]
 */
void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
	bool needCall = false;
	for (IOWrap::Output3DWrapper* ow : wraps)
		needCall = needCall || ow->needPushDepthImage();
	if (!needCall) return;

	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

	MinimalImageB3 iRImg(wl, hl);

	for (int i = 0; i < wl * hl; i++)
		iRImg.at(i) = Vec3b(colorRef[i][0], colorRef[i][0], colorRef[i][0]);


	int npts = numPoints[lvl];

	float nid = 0, sid = 0;
	for (int i = 0; i < npts; i++)
	{
		Pnt* point = points[lvl] + i;
		if (point->isGood)
		{
			nid++;
			sid += point->iR;
		}
	}
	float fac = nid / sid;
	for (int i = 0; i < npts; i++)
	{
		Pnt* point = points[lvl] + i;

		if (!point->isGood)
			iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, Vec3b(0, 0, 0));

		else
			iRImg.setPixel9(point->u + 0.5f, point->v + 0.5f, makeRainbow3B(point->iR * fac));
	}

	//IOWrap::displayImage("idepth-R", &iRImg, false);
	for (IOWrap::Output3DWrapper* ow : wraps)
		ow->pushDepthImage(&iRImg);
}

// calculates residual, Hessian and Hessian-block neede for re-substituting depth.
/**
 * [CoarseInitializer::calcResAndGS description]
 * @param  lvl          [description]
 * @param  H_out        [description]
 * @param  b_out        [description]
 * @param  H_out_sc     [description]
 * @param  b_out_sc     [description]
 * @param  refToNew     [description]
 * @param  refToNew_aff [description]
 * @param  plot         [description]
 * @return              [description]
 */
Vec3f CoarseInitializer::calcResAndGS(
  int lvl, Mat88f &H_out, Vec8f &b_out,
  Mat88f &H_out_sc, Vec8f &b_out_sc,
  SE3 refToNew, AffLight refToNew_aff,
  bool plot)
{
	//当前层的大小
	int wl = w[lvl], hl = h[lvl];
	//第一帧的图像和当前帧的图像
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
	Eigen::Vector3f* colorNew = newFrame->dIp[lvl];

	//当前帧与参考帧初始位姿变换，初始a和b
	//R=r*k
	Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
	//t
	Vec3f t = refToNew.translation().cast<float>();
	//e^a,b
	Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

	//当前层的内参
	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];

	Accumulator11 E;
	acc9.initialize();
	E.initialize();

	//第一帧每个点
	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	//遍历每一个点
	for (int i = 0; i < npts; i++)
	{
		Pnt* point = ptsl + i;
		//点的最大步骤，10的10次方
		point->maxstep = 1e10;
		//如果点不好，跳过这个点
		if (!point->isGood)
		{
			//则总能量加上这个点的能量，光度误差
			E.updateSingle((float)(point->energy[0]));
			//更新这个点的能量
			point->energy_new = point->energy;
			//点isGood和isGood_new都是false
			point->isGood_new = false;
			continue;
		}

		//8*1
		EIGEN_ALIGN16 VecNRf dp0;
		EIGEN_ALIGN16 VecNRf dp1;
		EIGEN_ALIGN16 VecNRf dp2;
		EIGEN_ALIGN16 VecNRf dp3;
		EIGEN_ALIGN16 VecNRf dp4;
		EIGEN_ALIGN16 VecNRf dp5;
		EIGEN_ALIGN16 VecNRf dp6;
		EIGEN_ALIGN16 VecNRf dp7;
		EIGEN_ALIGN16 VecNRf dd;
		EIGEN_ALIGN16 VecNRf r;
		//雅克比设为0
		JbBuffer_new[i].setZero();

		// sum over all residuals.
		// 残差和
		bool isGood = true;
		float energy = 0;
		//模式SSE,模式有8个点
		for (int idx = 0; idx < patternNum; idx++)
		{
			int dx = patternP[idx][0];
			int dy = patternP[idx][1];

			//ProjectPoint()
			//投影 host 上具有深度为 d p 点 (u h , v h ) 过程中,通过 R, t 变换后,对应的新的深度 d np 和投影点 Ku, Kv。
			//注意,计算过程中采用的是归一化深度
			//将第一帧中每个点根据上一次迭代的位姿投影到当前帧上
			//公式P'=R*P+t*dp
			Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) + t * point->idepth_new;
			//得到像素坐标u,v
			//公式P‘x/P'z,P'y/P'z
			float u = pt[0] / pt[2];
			float v = pt[1] / pt[2];
			//内参变换后的ku,kv
			//公式u*fx+cx,v*fy+cy
			float Ku = fxl * u + cxl;
			float Kv = fyl * v + cyl;
			//这个点深度除以当前得到的点深度 新的深度
			//公式dp'=dp/P’z
			float new_idepth = point->idepth_new / pt[2];

			//点是否在图像内且新的深度是否大于0
			if (!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && new_idepth > 0))
			{
				isGood = false;
				break;
			}

			//根据重投影点，进行双线性插值，获取在当前图像的灰度值，fx,fy
			Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
			//Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);
			//根据主导帧的点，进行双线性插值，获取在主导帧图像中的灰度值
			//float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
			float rlR = getInterpolatedElement31(colorRef, point->u + dx, point->v + dy, wl);

			//是否有值判断这两个值
			if (!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
			{
				isGood = false;
				break;
			}

			//计算光度误差残差 Ij-e^a*Ii-b
			float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
			//huber函数，setting_huberTH=9
			//小于阈值，则是1
			//大于阈值，则是阈值/res
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			//能量+=|Ij-e^a*Ii-b|^2huber
			//r^2		if |x|<k
			//th*r*(2-th/r) 	if |x|>k
			energy += hw * residual * residual * (2 - hw);

			//几何部分之深度 Jpdd
			//公式sd (t[0] − u*t[2])fx
			//sd (t[1] − v*t[2])fy
			float dxdd = (t[0] - t[2] * u) / pt[2];
			float dydd = (t[1] - t[2] * v) / pt[2];

			//若hw小于1,则hw开更号
			if (hw < 1) hw = sqrtf(hw);
			//几何部分之光照梯度 JIdx，就是常规的图像梯度图计算
			//当前图像的重投影后的点的x和y方向梯度，乘以fx,fy
			float dxInterp = hw * hitColor[1] * fxl;
			float dyInterp = hw * hitColor[2] * fyl;
			//公式JIdx = [dI(p ′ x ), dI(p ′ y) ]
			dp0[idx] = new_idepth * dxInterp;
			dp1[idx] = new_idepth * dyInterp;
			dp2[idx] = -new_idepth * (u * dxInterp + v * dyInterp);
			dp3[idx] = -u * v * dxInterp - (1 + v * v) * dyInterp;
			dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp;
			dp5[idx] = -v * dxInterp + u * dyInterp;
			dp6[idx] = - hw * r2new_aff[0] * rlR;
			dp7[idx] = - hw * 1;
			dd[idx] = dxInterp * dxdd  + dyInterp * dydd;

			//huber的残差
			r[idx] = hw * residual;

			//根据深度计算迭代步长
			float maxstep = 1.0f / Vec2f(dxdd * fxl, dydd * fyl).norm();

			//更新每个点的最大步长，若新的最大步长小于之前的最大步长，则更新这个点的最大步长
			if (maxstep < point->maxstep)
				point->maxstep = maxstep;

			// immediately compute dp*dd' and dd*dd' in JbBuffer1.
			// 更新JbBuffer_new
			JbBuffer_new[i][0] += dp0[idx] * dd[idx];
			JbBuffer_new[i][1] += dp1[idx] * dd[idx];
			JbBuffer_new[i][2] += dp2[idx] * dd[idx];
			JbBuffer_new[i][3] += dp3[idx] * dd[idx];
			JbBuffer_new[i][4] += dp4[idx] * dd[idx];
			JbBuffer_new[i][5] += dp5[idx] * dd[idx];
			JbBuffer_new[i][6] += dp6[idx] * dd[idx];
			JbBuffer_new[i][7] += dp7[idx] * dd[idx];
			JbBuffer_new[i][8] += r[idx] * dd[idx];
			JbBuffer_new[i][9] += dd[idx] * dd[idx];
		}
		//这次投影的点不好或者当前误差能量值大与点阈值*20
		if (!isGood || energy > point->outlierTH * 20)
		{
			//则总能量加上这个点的能量，光度误差,此时这个点的能量值为上一时刻迭代的值，初值为0
			E.updateSingle((float)(point->energy[0]));
			point->isGood_new = false;
			point->energy_new = point->energy;
			continue;
		}

		// 增加这个点8SSE模式下计算出的总energy
		// SSEData+=energy，num和numIn1++
		// add into energy.
		E.updateSingle(energy);
		//设这个点好的isGood_new=true，即在计算时用到了这个点
		point->isGood_new = true;
		//更新这个点的能量
		point->energy_new[0] = energy;

		//更新Hessian矩阵
		// update Hessian matrix.
		for (int i = 0; i + 3 < patternNum; i += 4)
			acc9.updateSSE(
			  _mm_load_ps(((float*)(&dp0)) + i),
			  _mm_load_ps(((float*)(&dp1)) + i),
			  _mm_load_ps(((float*)(&dp2)) + i),
			  _mm_load_ps(((float*)(&dp3)) + i),
			  _mm_load_ps(((float*)(&dp4)) + i),
			  _mm_load_ps(((float*)(&dp5)) + i),
			  _mm_load_ps(((float*)(&dp6)) + i),
			  _mm_load_ps(((float*)(&dp7)) + i),
			  _mm_load_ps(((float*)(&r)) + i));


		//SSE 8模式下，不进行这一步
		for (int i = ((patternNum >> 2) << 2); i < patternNum; i++)
			acc9.updateSingle(
			  (float)dp0[i], (float)dp1[i], (float)dp2[i], (float)dp3[i],
			  (float)dp4[i], (float)dp5[i], (float)dp6[i], (float)dp7[i],
			  (float)r[i]);
	}

	//总误差,得到A
	E.finish();
	//总的hessian矩阵
	acc9.finish();

	// calculate alpha energy, and decide if we cap it.
	// 计算alpha能量，然后决定我们是否正则项。
	// EAlpha？？？ EAlpha根本没计算，EAlpha.A一直=0
	Accumulator11 EAlpha;
	EAlpha.initialize();
	//遍历每一个点
	for (int i = 0; i < npts; i++)
	{
		Pnt* point = ptsl + i;
		//如果当前点不好，则不加
		if (!point->isGood_new)
		{
			E.updateSingle((float)(point->energy[1]));
		}
		//否则当前点的energy_new[1]=(idepth_new-1)^2，当前点深度平方
		//总误差能量值加上深度平方
		else
		{
			point->energy_new[1] = (point->idepth_new - 1) * (point->idepth_new - 1);
			E.updateSingle((float)(point->energy_new[1]));
		}
	}
	EAlpha.finish();
	//alphaW=150*150，权重*(每个点的逆深度和+平移*点数)
	//alphaEnergy？？  alphaEnergy=权重*(refToNew.translation().squaredNorm() * npts)
	float alphaEnergy = alphaW * (EAlpha.A + refToNew.translation().squaredNorm() * npts);

	//printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);
	// compute alpha opt.
	float alphaOpt;
	//alphaK=2.5*2.5
	//alphaEnergy>6.25*点数，则alphaOpt=0，alphaEnergy=6.25*点数
	//若平移超过alphaK，alphaOpt=0，且alphaEnergy=alphaK*npts
	//否则alphaOpt=alphaW，所以这里意思是refToNew.translation().squaredNorm()>150*150/2.5/2.5
	//与主导帧的平移量要大于60
	if (alphaEnergy > alphaK * npts)
	{
		alphaOpt = 0;
		alphaEnergy = alphaK * npts;
	}
	else
	{
		alphaOpt = alphaW;
	}

	acc9SC.initialize();
	for (int i = 0; i < npts; i++)
	{
		Pnt* point = ptsl + i;
		if (!point->isGood_new)
			continue;
		//更新每个点最新的雅克比
		point->lastHessian_new = JbBuffer_new[i][9];

		//雅克比最后两位
		JbBuffer_new[i][8] += alphaOpt * (point->idepth_new - 1);
		JbBuffer_new[i][9] += alphaOpt;

		//若这个权重为0,couplingWeight=1,则加上逆深度的变化
		if (alphaOpt == 0)
		{
			JbBuffer_new[i][8] += couplingWeight * (point->idepth_new - point->iR);
			JbBuffer_new[i][9] += couplingWeight;
		}

		JbBuffer_new[i][9] = 1 / (1 + JbBuffer_new[i][9]);
		//更新带权重的值
		acc9SC.updateSingleWeighted(
		  (float)JbBuffer_new[i][0], (float)JbBuffer_new[i][1], (float)JbBuffer_new[i][2], (float)JbBuffer_new[i][3],
		  (float)JbBuffer_new[i][4], (float)JbBuffer_new[i][5], (float)JbBuffer_new[i][6], (float)JbBuffer_new[i][7],
		  (float)JbBuffer_new[i][8], (float)JbBuffer_new[i][9]);
	}
	acc9SC.finish();

	//printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
	H_out = acc9.H.topLeftCorner<8, 8>(); // / acc9.num;
	b_out = acc9.H.topRightCorner<8, 1>(); // / acc9.num;
	H_out_sc = acc9SC.H.topLeftCorner<8, 8>(); // / acc9.num;
	b_out_sc = acc9SC.H.topRightCorner<8, 1>(); // / acc9.num;

	//若平移过大，则不加上
	H_out(0, 0) += alphaOpt * npts;
	H_out(1, 1) += alphaOpt * npts;
	H_out(2, 2) += alphaOpt * npts;

	//位姿变换的前3项，即平移的对数化
	Vec3f tlog = refToNew.log().head<3>().cast<float>();

	//若平移过大，则不加上
	//平移乘以点个数乘以权重
	b_out[0] += tlog[0] * alphaOpt * npts;
	b_out[1] += tlog[1] * alphaOpt * npts;
	b_out[2] += tlog[2] * alphaOpt * npts;

	//返回总误差能量值，带权重逆深度能量值，总误差能量数
	return Vec3f(E.A, alphaEnergy , E.num);
}

/**
 * [CoarseInitializer::rescale description]
 * @return [description]
 */
float CoarseInitializer::rescale()
{
	float factor = 20 * thisToNext.translation().norm();
//	float factori = 1.0f/factor;
//	float factori2 = factori*factori;
//
//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
//	{
//		int npts = numPoints[lvl];
//		Pnt* ptsl = points[lvl];
//		for(int i=0;i<npts;i++)
//		{
//			ptsl[i].iR *= factor;
//			ptsl[i].idepth_new *= factor;
//			ptsl[i].lastHessian *= factori2;
//		}
//	}
//	thisToNext.translation() *= factori;

	return factor;
}

/**
 * [CoarseInitializer::calcEC description]
 * @param  lvl [description]
 * @return     [description]
 * 计算总深度能量值函数E
 */
Vec3f CoarseInitializer::calcEC(int lvl)
{
	//snapped为false,即平移都还不够的时候,则返回0,0和点的个数个数
	if (!snapped) return Vec3f(0, 0, numPoints[lvl]);
	AccumulatorX<2> E;
	E.initialize();
	int npts = numPoints[lvl];
	//遍历每个点
	for (int i = 0; i < npts; i++)
	{
		//每个点
		Pnt* point = points[lvl] + i;
		//这个点是否够好
		if (!point->isGood_new) continue;
		//旧的深度值
		float rOld = (point->idepth - point->iR);
		//新的深度值
		float rNew = (point->idepth_new - point->iR);
		//加上无权重的旧的深度值平方，新的深度值平方
		E.updateNoWeight(Vec2f(rOld * rOld, rNew * rNew));

		//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
	}
	E.finish();

	//返回旧的深度和，新的深度和，点数
	//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
	return Vec3f(couplingWeight * E.A1m[0], couplingWeight * E.A1m[1], E.num);
}

/**
 * [CoarseInitializer::optReg description]
 * @param lvl [description]
 * 更新某一层的点的逆深度iR,iR为初始深度
 */
void CoarseInitializer::optReg(int lvl)
{
	//这层的点数
	int npts = numPoints[lvl];
	//这层的点
	Pnt* ptsl = points[lvl];
	//如果未初始化成功
	if (!snapped)
	{
		for (int i = 0; i < npts; i++)
			ptsl[i].iR = 1;
		return;
	}

	//则全设为1
	for (int i = 0; i < npts; i++)
	{
		Pnt* point = ptsl + i;
		if (!point->isGood) continue;

		//附近10个点的逆深度
		float idnn[10];
		//附近10个点中有几个好的点
		int nnn = 0;
		for (int j = 0; j < 10; j++)
		{
			if (point->neighbours[j] == -1) continue;
			Pnt* other = ptsl + point->neighbours[j];
			if (!other->isGood) continue;
			idnn[nnn] = other->iR;
			nnn++;
		}

		//若附近10个点中好的点数大于2
		if (nnn > 2)
		{
			//10个点逆深度中值
			std::nth_element(idnn, idnn + nnn / 2, idnn + nnn);
			//当前点的深度为当前这个点的深度与中值点深度互补
			point->iR = (1 - regWeight) * point->idepth + regWeight * idnn[nnn / 2];
		}
	}

}

/**
 * [CoarseInitializer::propagateUp description]
 * @param srcLvl [description]
 * 向上传递
 */
void CoarseInitializer::propagateUp(int srcLvl)
{
	assert(srcLvl + 1 < pyrLevelsUsed);
	// set idepth of target

	int nptss = numPoints[srcLvl];
	int nptst = numPoints[srcLvl + 1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl + 1];

	// set to zero.
	for (int i = 0; i < nptst; i++)
	{
		Pnt* parent = ptst + i;
		parent->iR = 0;
		parent->iRSumNum = 0;
	}

	for (int i = 0; i < nptss; i++)
	{
		Pnt* point = ptss + i;
		if (!point->isGood) continue;

		Pnt* parent = ptst + point->parent;
		parent->iR += point->iR * point->lastHessian;
		parent->iRSumNum += point->lastHessian;
	}

	for (int i = 0; i < nptst; i++)
	{
		Pnt* parent = ptst + i;
		if (parent->iRSumNum > 0)
		{
			parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
			parent->isGood = true;
		}
	}

	//更新上一层的点的逆深度iR
	optReg(srcLvl + 1);
}

/**
 * [CoarseInitializer::propagateDown description]
 * @param srcLvl [description]
 * 向下传递
 */
void CoarseInitializer::propagateDown(int srcLvl)
{
	assert(srcLvl > 0);
	// set idepth of target

	//下一层点的个数
	int nptst = numPoints[srcLvl - 1];
	//当前层的点
	Pnt* ptss = points[srcLvl];
	//下一层的点
	Pnt* ptst = points[srcLvl - 1];

	for (int i = 0; i < nptst; i++)
	{
		//下一层的点
		Pnt* point = ptst + i;
		//下一层点的父点
		Pnt* parent = ptss + point->parent;

		//当前层这个点不成功或者这个Hessian值过小，则跳过
		if (!parent->isGood || parent->lastHessian < 0.1) continue;
		if (!point->isGood)
		{
			//下一层的这个点的逆深度=当前层
			point->iR = point->idepth = point->idepth_new = parent->iR;
			//这个点设为可以迭代
			point->isGood = true;
			point->lastHessian = 0;
		}
		else
		{
			//新的iR=(下一层的iR*lastH*2+当前层的iR*lastH)/(下一层lastH*2+当前层的lastH)
			float newiR = (point->iR * point->lastHessian * 2 + parent->iR * parent->lastHessian) / (point->lastHessian * 2 + parent->lastHessian);
			//这个点逆深度
			point->iR = point->idepth = point->idepth_new = newiR;
		}
	}

	//更新下一层的点的逆深度iR
	optReg(srcLvl - 1);
}

/**
 * [CoarseInitializer::makeGradients description]
 * @param data [description]
 * 设置梯度
 */
void CoarseInitializer::makeGradients(Eigen::Vector3f** data)
{
	for (int lvl = 1; lvl < pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl - 1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		Eigen::Vector3f* dINew_l = data[lvl];
		Eigen::Vector3f* dINew_lm = data[lvlm1];

		for (int y = 0; y < hl; y++)
			for (int x = 0; x < wl; x++)
				dINew_l[x + y * wl][0] = 0.25f * (dINew_lm[2 * x   + 2 * y * wlm1][0] +
				                                  dINew_lm[2 * x + 1 + 2 * y * wlm1][0] +
				                                  dINew_lm[2 * x   + 2 * y * wlm1 + wlm1][0] +
				                                  dINew_lm[2 * x + 1 + 2 * y * wlm1 + wlm1][0]);

		for (int idx = wl; idx < wl * (hl - 1); idx++)
		{
			dINew_l[idx][1] = 0.5f * (dINew_l[idx + 1][0] - dINew_l[idx - 1][0]);
			dINew_l[idx][2] = 0.5f * (dINew_l[idx + wl][0] - dINew_l[idx - wl][0]);
		}
	}
}

// set first frame
/**
 * [CoarseInitializer::setFirstStereo description]
 * @param HCalib                [description]
 * @param newFrameHessian       [description]
 * @param newFrameHessian_Right [description]
 */
void CoarseInitializer::setFirstStereo(CalibHessian* HCalib, FrameHessian* newFrameHessian, FrameHessian* newFrameHessian_Right)
{
	//设置内参
	makeK(HCalib);
	//第一帧和右帧
	firstFrame = newFrameHessian;
	firstRightFrame = newFrameHessian_Right;

	//点选择器
	PixelSelector sel(w[0], h[0]);

	//第一层的每个点的选择图
	float* statusMap = new float[w[0]*h[0]];
	//后面层
	bool* statusMapB = new bool[w[0]*h[0]];

	Mat33f K = Mat33f::Identity();
	K(0, 0) = HCalib->fxl();
	K(1, 1) = HCalib->fyl();
	K(0, 2) = HCalib->cxl();
	K(1, 2) = HCalib->cyl();

	float densities[] = {0.03, 0.05, 0.15, 0.5, 1};
	memset(idepth[0], 0, sizeof(float)*w[0]*h[0]);

	//为每一层选择点,statusMap!=0　statusMapB!=0，则是选择的点
	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
	{
		sel.currentPotential = 3;
		int npts, npts_right;
		if (lvl == 0)
		{
			npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * w[0] * h[0], 1, false, 2);
		}
		else
		{
            npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl] * w[0] * h[0]);
		}

		if (points[lvl] != 0) delete[] points[lvl];
		points[lvl] = new Pnt[npts];

		// set idepth map by static stereo matching. if no idepth is available, set 0.01.
		int wl = w[lvl], hl = h[lvl];
		//该层的点
		Pnt* pl = points[lvl];
		int nl = 0;
		for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++)
			for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++)
			{
				if (lvl == 0 && statusMap[x + y * wl] != 0)
				{
					ImmaturePoint* pt = new ImmaturePoint(x, y, firstFrame, statusMap[x + y * wl], HCalib);

					pt->u_stereo = pt->u;
					pt->v_stereo = pt->v;
					pt->idepth_min_stereo = 0;
					pt->idepth_max_stereo = NAN;

					//静态双目匹配，左图与右图进行匹配
					ImmaturePointStatus stat = pt->traceStereo(firstRightFrame, K, 1);

					if (stat == ImmaturePointStatus::IPS_GOOD)
					{
						//			    	assert(patternNum==9);
						pl[nl].u = x;
						pl[nl].v = y;

						//点额逆深度
						pl[nl].idepth = pt->idepth_stereo;
						pl[nl].iR = pt->idepth_stereo;

						//点的状态
						pl[nl].isGood = true;
						pl[nl].energy.setZero();
						pl[nl].lastHessian = 0;
						pl[nl].lastHessian_new = 0;
						pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];
						idepth[0][x + wl * y] = pt->idepth_stereo;

						//点的灰度值和梯度值
						Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y * w[lvl];

						//模式和
						float sumGrad2 = 0;
						for (int idx = 0; idx < patternNum; idx++)
						{
							int dx = patternP[idx][0];
							int dy = patternP[idx][1];
							float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
							sumGrad2 += absgrad;
						}

						//点的out阈值
						pl[nl].outlierTH = patternNum * setting_outlierTH;
						//点++
						nl++;
						assert(nl <= npts);
					}
					else
					{
						pl[nl].u = x;
						pl[nl].v = y;
						pl[nl].idepth = 0.01;
						//printf("the idepth is: %f\n", pl[nl].idepth);
						pl[nl].iR = 0.01;
						pl[nl].isGood = true;
						pl[nl].energy.setZero();
						pl[nl].lastHessian = 0;
						pl[nl].lastHessian_new = 0;
						pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];
						idepth[0][x + wl * y] = 0.01;

						Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y * w[lvl];
						float sumGrad2 = 0;
						for (int idx = 0; idx < patternNum; idx++)
						{
							int dx = patternP[idx][0];
							int dy = patternP[idx][1];
							float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
							sumGrad2 += absgrad;
						}

						pl[nl].outlierTH = patternNum * setting_outlierTH;

						nl++;
						assert(nl <= npts);
					}

					delete pt;
				}

				//后面层的点
				if (lvl != 0 && statusMapB[x + y * wl])
				{
					int lvlm1 = lvl - 1;
					int wlm1 = w[lvlm1];
					float* idepth_l = idepth[lvl];
					float* idepth_lm = idepth[lvlm1];
					//assert(patternNum==9);
					pl[nl].u = x + 0.1;
					pl[nl].v = y + 0.1;
					//点的逆深度设为１,
					pl[nl].idepth = 1;
					pl[nl].iR = 1;
					pl[nl].isGood = true;
					pl[nl].energy.setZero();
					pl[nl].lastHessian = 0;
					pl[nl].lastHessian_new = 0;
					pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];
					int bidx = 2 * x   + 2 * y * wlm1;
					idepth_l[x + y * wl] = idepth_lm[bidx] +
					                       idepth_lm[bidx + 1] +
					                       idepth_lm[bidx + wlm1] +
					                       idepth_lm[bidx + wlm1 + 1];

					Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y * w[lvl];
					float sumGrad2 = 0;
					for (int idx = 0; idx < patternNum; idx++)
					{
						int dx = patternP[idx][0];
						int dy = patternP[idx][1];
						float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
						sumGrad2 += absgrad;
					}

					pl[nl].outlierTH = patternNum * setting_outlierTH;

					nl++;
					assert(nl <= npts);
				}
			}

		//每一层的点数
		numPoints[lvl] = nl;
	}

	delete[] statusMap;
	delete[] statusMapB;

	//
	makeNN();

	thisToNext = SE3();
	snapped = false;
	frameID = snappedAt = 0;

	for (int i = 0; i < pyrLevelsUsed; i++)
		dGrads[i].setZero();

}

/**
 * [CoarseInitializer::setFirst description]
 * @param HCalib          [description]
 * @param newFrameHessian [description]
 */
void CoarseInitializer::setFirst(	CalibHessian* HCalib, FrameHessian* newFrameHessian)
{

	makeK(HCalib);
	firstFrame = newFrameHessian;

	PixelSelector sel(w[0], h[0]);

	float* statusMap = new float[w[0]*h[0]];
	bool* statusMapB = new bool[w[0]*h[0]];

	//每一层的点的数量，越大的图像点越少
	float densities[] = {0.03, 0.05, 0.15, 0.5, 1};
	//从最大的层向上
	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
	{
		//当前候选
		sel.currentPotential = 3;
		int npts;
		//选点
		if (lvl == 0)
			npts = sel.makeMaps(firstFrame, statusMap, densities[lvl] * w[0] * h[0], 1, false, 2);
		else
			npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl] * w[0] * h[0]);

		if (points[lvl] != 0) delete[] points[lvl];
		points[lvl] = new Pnt[npts];

		// set idepth map to initially 1 everywhere.
		int wl = w[lvl], hl = h[lvl];
		Pnt* pl = points[lvl];
		int nl = 0;
		for (int y = patternPadding + 1; y < hl - patternPadding - 2; y++)
			for (int x = patternPadding + 1; x < wl - patternPadding - 2; x++)
			{
				//if(x==2) printf("y=%d!\n",y);
				if (lvl != 0 && statusMapB[x + y * wl])
				{
					//assert(patternNum==9);
					//点的u,v
					pl[nl].u = x + 0.1;
					pl[nl].v = y + 0.1;
					//点的逆深度
					pl[nl].idepth = 1;
					//点的iR
					pl[nl].iR = 1;
					//点是初始都是好的
					pl[nl].isGood = true;
					//每个点的初始能量都是0
					pl[nl].energy.setZero();
					//每个点Hessian
					pl[nl].lastHessian = 0;
					pl[nl].lastHessian_new = 0;
					//点的类型
					pl[nl].my_type = (lvl != 0) ? 1 : statusMap[x + y * wl];

					Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y * w[lvl];
					//这个点在模式中的梯度和
					float sumGrad2 = 0;
					for (int idx = 0; idx < patternNum; idx++)
					{
						int dx = patternP[idx][0];
						int dy = patternP[idx][1];
						//x方向和y方向梯度平方和
						float absgrad = cpt[dx + dy * w[lvl]].tail<2>().squaredNorm();
						sumGrad2 += absgrad;
					}

//				float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
//				pl[nl].outlierTH = patternNum*gth*gth;
//
					//这个点剔除的阈值
					pl[nl].outlierTH = patternNum * setting_outlierTH;

					nl++;
					assert(nl <= npts);
				}
			}
		//这一层的点数
		numPoints[lvl] = nl;
	}
	delete[] statusMap;
	delete[] statusMapB;

	//knn
	makeNN();

	//当前与下一阵的变换初始化
	thisToNext = SE3();
	snapped = false;
	//初始话的帧id
	frameID = snappedAt = 0;

	for (int i = 0; i < pyrLevelsUsed; i++)
		dGrads[i].setZero();
}

/**
 * [CoarseInitializer::resetPoints description]
 * @param lvl [description]
 */
void CoarseInitializer::resetPoints(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for (int i = 0; i < npts; i++)
	{
		//每个点能量设为0
		pts[i].energy.setZero();
		//每个点新的逆深度先设为原来的逆深度
		pts[i].idepth_new = pts[i].idepth;
		//等于最上层或者这个点不成功,
		//设置每个点的初始iR，idepth，idepth_new
		if (lvl == pyrLevelsUsed - 1 && !pts[i].isGood)
		{
			float snd = 0, sn = 0;
			//将其周围10个点的iR相加
			for (int n = 0; n < 10; n++)
			{
				if (pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue;
				snd += pts[pts[i].neighbours[n]].iR;
				sn += 1;
			}

			if (sn > 0)
			{
				//周围有号的点
				pts[i].isGood = true;
				//则iR ,逆深度和新的逆深度为周围平均
				pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd / sn;
			}
		}
	}
}

/**
 * [CoarseInitializer::doStep description]
 * @param lvl    [description]
 * @param lambda [description]
 * @param inc    [description]
 * 下一步,更新每个点的idepth_new
 */
void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
{
	//最大像素步数
	const float maxPixelStep = 0.25;
	//最大id步数
	const float idMaxStep = 1e10;

	//点
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];

	//遍历每一个点
	for (int i = 0; i < npts; i++)
	{
		//这个点不好
		if (!pts[i].isGood) continue;

		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
		float step = - b * JbBuffer[i][9] / (1 + lambda);

		//每个点的步长*最大像素步长
		float maxstep = maxPixelStep * pts[i].maxstep;
		//设置最大步长
		if (maxstep > idMaxStep) maxstep = idMaxStep;

		//限制步长
		if (step >  maxstep) step = maxstep;
		if (step < -maxstep) step = -maxstep;

		//新的逆深度
		float newIdepth = pts[i].idepth + step;
		if (newIdepth < 1e-3 ) newIdepth = 1e-3;
		if (newIdepth > 50) newIdepth = 50;

		//设置点的新的深度idepth_new
		pts[i].idepth_new = newIdepth;
	}
}

/**
 * [CoarseInitializer::applyStep description]
 * @param lvl [description]
 * 更新这一层每个点的数据
 */
void CoarseInitializer::applyStep(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for (int i = 0; i < npts; i++)
	{
		//当前这个点不好
		if (!pts[i].isGood)
		{
			//将当前这个点的逆深度设为初值iR
			pts[i].idepth = pts[i].idepth_new = pts[i].iR;
			continue;
		}
		//更新这个点的能量函数，是否good，逆深度，最新的Hessian矩阵
		pts[i].energy = pts[i].energy_new;
		pts[i].isGood = pts[i].isGood_new;
		pts[i].idepth = pts[i].idepth_new;
		pts[i].lastHessian = pts[i].lastHessian_new;
	}

	//将雅克比进行更新
	std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

/**
 * [CoarseInitializer::makeK description]
 * @param HCalib [description]
 */
void CoarseInitializer::makeK(CalibHessian* HCalib)
{
	//原始图像大小
	w[0] = wG[0];
	h[0] = hG[0];

	//内参
	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	//设置每一层的内参K
	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level - 1] * 0.5;
		fy[level] = fy[level - 1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1 << level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1 << level) - 0.5;
	}

	//设置每一层的内参的逆
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
 * [CoarseInitializer::makeNN description]
 * 找到每一层中的每个点周围的10个点
 * 除了最上层外的几层中的点的父点
 */
void CoarseInitializer::makeNN()
{
	//最近邻的距离
	const float NNDistFactor = 0.05;

	//构建kdtree
	typedef nanoflann::KDTreeSingleIndexAdaptor <
	nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,
	          FLANNPointcloud, 2 > KDTree;

	// build indices
	// 多层的FLANN金字塔
	FLANNPointcloud pcs[PYR_LEVELS];
	KDTree* indexes[PYR_LEVELS];
	//从下往上
	for (int i = 0; i < pyrLevelsUsed; i++)
	{
		//构建每一层点云
		pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
		//构建这一层的kd数
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5) );
		indexes[i]->buildIndex();
	}

	//寻周周围10个点
	const int nn = 10;

	//寻找最近点和父点
	// find NN & parents
	for (int lvl = 0; lvl < pyrLevelsUsed; lvl++)
	{
		//当前点
		Pnt* pts = points[lvl];
		//当前点数
		int npts = numPoints[lvl];

		int ret_index[nn];
		float ret_dist[nn];
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		for (int i = 0; i < npts; i++)
		{
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			resultSet.init(ret_index, ret_dist);
			Vec2f pt = Vec2f(pts[i].u, pts[i].v);
			//寻找这个点周围10个点
			indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());
			int myidx = 0;
			float sumDF = 0;
			//遍历这10个点
			for (int k = 0; k < nn; k++)
			{
				//设置这个点周围的10个点
				pts[i].neighbours[myidx] = ret_index[k];
				float df = expf(-ret_dist[k] * NNDistFactor);
				sumDF += df;
				//距离
				pts[i].neighboursDist[myidx] = df;
				assert(ret_index[k] >= 0 && ret_index[k] < npts);
				myidx++;
			}
			//与周围每个点的距离都是相等的？，10/sumDF
			for (int k = 0; k < nn; k++)
				pts[i].neighboursDist[k] *= 10 / sumDF;

			//不是最上层
			if (lvl < pyrLevelsUsed - 1 )
			{
				resultSet1.init(ret_index, ret_dist);
				//当前点缩小
				pt = pt * 0.5f - Vec2f(0.25f, 0.25f);
				//在上一层中找到这个点的最近的点
				indexes[lvl + 1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());

				//设置这个点上层的父点
				pts[i].parent = ret_index[0];
				//与父点的距离
				pts[i].parentDist = expf(-ret_dist[0] * NNDistFactor);

				assert(ret_index[0] >= 0 && ret_index[0] < numPoints[lvl + 1]);
			}
			else
			{
				//最上层无父点
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}
	}
	for (int i = 0; i < pyrLevelsUsed; i++)
		delete indexes[i];
}
}

