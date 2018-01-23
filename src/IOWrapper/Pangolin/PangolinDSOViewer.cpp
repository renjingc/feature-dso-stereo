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



#include "PangolinDSOViewer.h"
#include "KeyFrameDisplay.h"

#include "util/settings.h"
#include "util/globalCalib.h"
#include "util/DatasetReader.h"
#include "FullSystem/FrameHessian.h"
#include "FullSystem/PointHessian.h"
#include "FullSystem/CalibHessian.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/ImmaturePoint.h"

#include <iostream>
#include <fstream>
#include <string>

namespace fdso
{
namespace IOWrap
{

PangolinDSOViewer::PangolinDSOViewer(int w, int h, std::string _gtPath,bool startRunThread):
	gtPath(_gtPath)
{
	//图像搜小后大小
	this->w = w;
	this->h = h;
	//运行
	running = true;

	{
		//图像上锁
		boost::unique_lock<boost::mutex> lk(openImagesMutex);
		internalVideoImg = new MinimalImageB3(w, h);
		internalKFImg = new MinimalImageB3(w, h);
		internalResImg = new MinimalImageB3(w, h);
		internalVideoImg_Right = new MinimalImageB3(w, h);

		videoImgChanged = kfImgChanged = resImgChanged = true;

		//初始化大小
		internalVideoImg->setBlack();
		internalVideoImg_Right->setBlack();
		internalKFImg->setBlack();
		internalResImg->setBlack();
	}

	{
		//当前相机
		currentCam = new KeyFrameDisplay();
	}

	//是否重置
	needReset = false;

	//显示线程
	if (startRunThread)
		runThread = boost::thread(&PangolinDSOViewer::run, this);
}

/**
 * @brief      Destroys the object.
 */
PangolinDSOViewer::~PangolinDSOViewer()
{
	close();
	runThread.join();
}

/**
 * @brief      { function_description }
 * 显示的住函数
 */
void PangolinDSOViewer::run()
{
	printf("START PANGOLIN!\n");

	//创建一个窗口,2倍的图像大小
	pangolin::CreateWindowAndBind("Main", 2 * w, 2 * h);
	const int UI_WIDTH = 180;
	//启动深度测试
	glEnable(GL_DEPTH_TEST);

	// 3D visualization
	//Define Projection and initial ModelView matrix
	pangolin::OpenGlRenderState Visualization3D_camera(
	  pangolin::ProjectionMatrix(w, h, 400, 400, w / 2, h / 2, 0.1, 1000),
	  //对应的是gluLookAt,摄像机位置,参考点位置,up vector(上向量)
	  pangolin::ModelViewLookAt(-0, -5, -10, 0, 0, 0, pangolin::AxisNegY)
	);

	// Create Interactive View in window
	//3d显示
	pangolin::View& Visualization3D_display = pangolin::CreateDisplay()
	    .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0, -w / (float)h)
	    .SetHandler(new pangolin::Handler3D(Visualization3D_camera));


	// 4 images
	//显示深度图
	pangolin::View& d_kfDepth = pangolin::Display("imgKFDepth")
	                            .SetAspect(w / (float)h);

	//显示右图的深度图
	pangolin::View& d_video_Right = pangolin::Display("imgKFDepth_Right")
	                                .SetAspect(w / (float)h);

	//显示视频
	pangolin::View& d_video = pangolin::Display("imgVideo")
	                          .SetAspect(w / (float)h);

	//显示残差
	pangolin::View& d_residual = pangolin::Display("imgResidual")
	                             .SetAspect(w / (float)h);

	pangolin::GlTexture texKFDepth(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
	pangolin::GlTexture texVideo(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
	pangolin::GlTexture texVideo_Right(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
	pangolin::GlTexture texResidual(w, h, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

	//创建显示
	pangolin::CreateDisplay()
	.SetBounds(0.0, 0.3, pangolin::Attach::Pix(UI_WIDTH), 1.0)
	.SetLayout(pangolin::LayoutEqual)
	.AddDisplay(d_kfDepth)
	.AddDisplay(d_video)
	.AddDisplay(d_video_Right)
	.AddDisplay(d_residual);

	// parameter reconfigure gui
	//创建参数调整器界面
	pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(UI_WIDTH));

	//模式
	pangolin::Var<int> settings_pointCloudMode("ui.PC_mode", 1, 1, 4, false);

	//是否显示关键帧
	pangolin::Var<bool> settings_showKFCameras("ui.KFCam", true, true);
	//显示当前相机位姿
	pangolin::Var<bool> settings_showCurrentCamera("ui.CurrCam", true, true);
	//显示关键路径
	pangolin::Var<bool> settings_showGroundTrajectory("ui.GroundTrajectory", false, true);
	//显示全部路径
	pangolin::Var<bool> settings_showFullTrajectory("ui.FullTrajectory", false, true);
	//显示连接
	pangolin::Var<bool> settings_showActiveConstraints("ui.ActiveConst", true, true);
	//显示所有的约束
	pangolin::Var<bool> settings_showAllConstraints("ui.AllConst", false, true);

	//3d显示
	pangolin::Var<bool> settings_show3D("ui.show3D", true, true);
	//深度图显示
	pangolin::Var<bool> settings_showLiveDepth("ui.showDepth", true, true);
	//显示视频
	pangolin::Var<bool> settings_showLiveVideo("ui.showVideo", true, true);
	//显示残差
	pangolin::Var<bool> settings_showLiveResidual("ui.showResidual", false, true);

	//显示滑动窗口
	pangolin::Var<bool> settings_showFramesWindow("ui.showFramesWindow", false, true);
	//显示所有的tracking
	pangolin::Var<bool> settings_showFullTracking("ui.showFullTracking", false, true);
	//显示粗跟踪
	pangolin::Var<bool> settings_showCoarseTracking("ui.showCoarseTracking", false, true);

	//参数
	pangolin::Var<int> settings_sparsity("ui.sparsity", 1, 1, 20, false);
	pangolin::Var<double> settings_scaledVarTH("ui.relVarTH", 0.001, 1e-10, 1e10, true);
	pangolin::Var<double> settings_absVarTH("ui.absVarTH", 0.001, 1e-10, 1e10, true);
	pangolin::Var<double> settings_minRelBS("ui.minRelativeBS", 0.1, 0, 1, false);

	//是否重置
	pangolin::Var<bool> settings_resetButton("ui.Reset", false, false);

	pangolin::Var<int> settings_nPts("ui.activePoints", setting_desiredPointDensity, 50, 5000, false);
	pangolin::Var<int> settings_nCandidates("ui.pointCandidates", setting_desiredImmatureDensity, 50, 5000, false);
	pangolin::Var<int> settings_nMaxFrames("ui.maxFrames", setting_maxFrames, 4, 10, false);
	pangolin::Var<double> settings_kfFrequency("ui.kfFrequency", setting_kfGlobalWeight, 0.1, 3, false);
	pangolin::Var<double> settings_gradHistAdd("ui.minGradAdd", setting_minGradHistAdd, 0, 15, false);

	//跟踪帧率
	pangolin::Var<double> settings_trackFps("ui.Track fps", 0, 0, 0, false);
	//关键帧率
	pangolin::Var<double> settings_mapFps("ui.KF fps", 0, 0, 0, false);

	// show ground truth
	//实际路线

	std::ifstream ReadFile(gtPath.c_str());
	std::string temp;
	std::string delim (" ");
	std::vector<std::string> results;
	Sophus::Matrix4f gtCam;
	//实际值
	std::vector<Sophus::Matrix4f> matrix_result;

	//读取实际路线
	while (std::getline(ReadFile, temp))
	{
		split(temp, delim, results);
		for (int i = 0; i < 3; i++)
			for (int j = 0; j < 4; j++)
			{
				gtCam(i, j) = atof(results[4 * i + j].c_str());
			}
		gtCam(3, 0) = 0;
		gtCam(3, 1) = 0;
		gtCam(3, 2) = 0;
		gtCam(3, 3) = 1;

		results.clear();
		matrix_result.push_back(gtCam);
	}

	ReadFile.close();

	//黄色
	float yellow[3] = {1, 1, 0};

	// Default hooks for exiting (Esc) and fullscreen (tab).
	while ( !pangolin::ShouldQuit() && running )
	{
		// Clear entire screen
		//清空窗口
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//显示3d画面
		if (setting_render_display3D)
		{
			// Activate efficiently by object
			Visualization3D_display.Activate(Visualization3D_camera);
			boost::unique_lock<boost::mutex> lk3d(model3DMutex);
			//pangolin::glDrawColouredCube();
			int refreshed = 0;
			//遍历全部关键帧
			for (KeyFrameDisplay* fh : keyframes)
			{
				float blue[3] = {0, 0, 1};
				float red[3] = {1, 0, 0};
				float yellow[3] = {0, 1, 0};
				//是否显示关键帧
				if (this->settings_showKFCameras)
					fh->drawCam(1, red, 0.1, false);

				refreshed = + (int)(fh->refreshPC(refreshed < 10, this->settings_scaledVarTH, this->settings_absVarTH,
				                                  this->settings_pointCloudMode, this->settings_minRelBS, this->settings_sparsity));
				//画每个点
				fh->drawPC(1);
			}

			//画真值路径
			if (this->settings_showGroundTrajectory && keyframes.size()>1)
			{
				for (int i = 0; i < matrix_result.size(); i++)
				{
					// KeyFrameDisplay* fh = new KeyFrameDisplay;
					keyframes[0]->drawGTCam(matrix_result[i], 3, yellow, 0.1);
					// delete(fh);
				}
			}

			//显示当前相机位姿
			if (this->settings_showCurrentCamera && currentCam && keyframes.size() > 1)
				currentCam->drawCam(2, 0, 0.2, true);

			//画连接
			drawConstraints();
			lk3d.unlock();
		}


		openImagesMutex.lock();
		//加载图像
		if (videoImgChanged) {
			texVideo.Upload(internalVideoImg->data, GL_BGR, GL_UNSIGNED_BYTE);
			texVideo_Right.Upload(internalVideoImg_Right->data, GL_BGR, GL_UNSIGNED_BYTE);
		}
		//加载深度图
		if (kfImgChanged) {
			texKFDepth.Upload(internalKFImg->data, GL_BGR, GL_UNSIGNED_BYTE);
		}
		//加载改变的残差图
		if (resImgChanged)
			texResidual.Upload(internalResImg->data, GL_BGR, GL_UNSIGNED_BYTE);
		videoImgChanged = kfImgChanged = resImgChanged = false;
		openImagesMutex.unlock();

		// update fps counters
		{
			openImagesMutex.lock();
			float sd = 0;
			for (float d : lastNMappingMs) sd += d;
			settings_mapFps = lastNMappingMs.size() * 1000.0f / sd;
			openImagesMutex.unlock();
		}
		{
			model3DMutex.lock();
			float sd = 0;
			for (float d : lastNTrackingMs) sd += d;
			settings_trackFps = lastNTrackingMs.size() * 1000.0f / sd;
			model3DMutex.unlock();
		}

		if (setting_render_displayVideo)
		{
			d_video.Activate();
			glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
			texVideo.RenderToViewportFlipY();

			d_video_Right.Activate();
			glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
			texVideo_Right.RenderToViewportFlipY();
		}

		if (setting_render_displayDepth)
		{
			d_kfDepth.Activate();
			glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
			texKFDepth.RenderToViewportFlipY();
		}

		if (setting_render_displayResidual)
		{
			d_residual.Activate();
			glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
			texResidual.RenderToViewportFlipY();
		}

		// update parameters
		//更新参数
		this->settings_pointCloudMode = settings_pointCloudMode.Get();

		this->settings_showActiveConstraints = settings_showActiveConstraints.Get();
		this->settings_showAllConstraints = settings_showAllConstraints.Get();
		this->settings_showCurrentCamera = settings_showCurrentCamera.Get();
		this->settings_showKFCameras = settings_showKFCameras.Get();
		this->settings_showGroundTrajectory = settings_showGroundTrajectory.Get();
		this->settings_showFullTrajectory = settings_showFullTrajectory.Get();

		setting_render_display3D = settings_show3D.Get();
		setting_render_displayDepth = settings_showLiveDepth.Get();
		setting_render_displayVideo =  settings_showLiveVideo.Get();
		setting_render_displayResidual = settings_showLiveResidual.Get();

		setting_render_renderWindowFrames = settings_showFramesWindow.Get();
		setting_render_plotTrackingFull = settings_showFullTracking.Get();
		setting_render_displayCoarseTrackingFull = settings_showCoarseTracking.Get();

		this->settings_absVarTH = settings_absVarTH.Get();
		this->settings_scaledVarTH = settings_scaledVarTH.Get();
		this->settings_minRelBS = settings_minRelBS.Get();
		this->settings_sparsity = settings_sparsity.Get();

		//====================TO DO : I set here.. not flexible===================
		setting_desiredPointDensity = settings_nPts.Get();
		setting_desiredImmatureDensity = settings_nCandidates.Get();

		setting_maxFrames = settings_nMaxFrames.Get();
		setting_kfGlobalWeight = settings_kfFrequency.Get();
		setting_minGradHistAdd = settings_gradHistAdd.Get();

		if (settings_resetButton.Get())
		{
			printf("RESET!\n");
			settings_resetButton.Reset();
			setting_fullResetRequested = true;
		}

		// Swap frames and Process Events
		pangolin::FinishFrame();

		if (needReset) reset_internal();
	}


	printf("QUIT Pangolin thread!\n");
	printf("I'll just kill the whole process.\nSo Long, and Thanks for All the Fish!\n");

	exit(1);
}


void PangolinDSOViewer::close()
{
	running = false;
}

void PangolinDSOViewer::join()
{
	runThread.join();
	printf("JOINED Pangolin thread!\n");
}

void PangolinDSOViewer::reset()
{
	needReset = true;
}

void PangolinDSOViewer::reset_internal()
{
	model3DMutex.lock();
	for (size_t i = 0; i < keyframes.size(); i++) delete keyframes[i];
	keyframes.clear();
	allFramePoses.clear();
	keyframesByKFID.clear();
	connections.clear();
	model3DMutex.unlock();


	openImagesMutex.lock();
	internalVideoImg->setBlack();
	internalVideoImg_Right->setBlack();
	internalKFImg->setBlack();
	internalResImg->setBlack();
	videoImgChanged = kfImgChanged = resImgChanged = true;
	openImagesMutex.unlock();

	needReset = false;
}

/**
 * @brief      Draws constraints.
 */
void PangolinDSOViewer::drawConstraints()
{
	//显示所有连接
	if (settings_showAllConstraints)
	{
		// draw constraints
		glLineWidth(1);
		glBegin(GL_LINES);

		glColor3f(0, 1, 0);
		glBegin(GL_LINES);
		for (unsigned int i = 0; i < connections.size(); i++)
		{
			if (connections[i].to == 0 || connections[i].from == 0) continue;
			int nAct = connections[i].bwdAct + connections[i].fwdAct;
			int nMarg = connections[i].bwdMarg + connections[i].fwdMarg;
			if (nAct == 0 && nMarg > 0  )
			{
				Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
				glVertex3f((GLfloat) t[0], (GLfloat) t[1], (GLfloat) t[2]);
				t = connections[i].to->camToWorld.translation().cast<float>();
				glVertex3f((GLfloat) t[0], (GLfloat) t[1], (GLfloat) t[2]);
			}
		}
		glEnd();
	}

	//显示激活的连接
	if (settings_showActiveConstraints)
	{
		glLineWidth(3);
		glColor3f(0, 0, 1);
		glBegin(GL_LINES);
		for (unsigned int i = 0; i < connections.size(); i++)
		{
			if (connections[i].to == 0 || connections[i].from == 0) continue;
			int nAct = connections[i].bwdAct + connections[i].fwdAct;

			if (nAct > 0)
			{
				Sophus::Vector3f t = connections[i].from->camToWorld.translation().cast<float>();
				glVertex3f((GLfloat) t[0], (GLfloat) t[1], (GLfloat) t[2]);
				t = connections[i].to->camToWorld.translation().cast<float>();
				glVertex3f((GLfloat) t[0], (GLfloat) t[1], (GLfloat) t[2]);
			}
		}
		glEnd();
	}

	// //显示关键帧路径
	// if (settings_showTrajectory)
	// {
	// 	float colorRed[3] = {1, 0, 0};
	// 	glColor3f(colorRed[0], colorRed[1], colorRed[2]);
	// 	glLineWidth(3);

	// 	glBegin(GL_LINE_STRIP);
	// 	for (unsigned int i = 0; i < keyframes.size(); i++)
	// 	{
	// 		glVertex3f((float)keyframes[i]->camToWorld.translation()[0],
	// 		           (float)keyframes[i]->camToWorld.translation()[1],
	// 		           (float)keyframes[i]->camToWorld.translation()[2]);
	// 	}
	// 	glEnd();
	// }

	//显示所有路径
	if (settings_showFullTrajectory)
	{
		float colorGreen[3] = {0, 1, 0};
		glColor3f(colorGreen[0], colorGreen[1], colorGreen[2]);
		glLineWidth(3);

		glBegin(GL_LINE_STRIP);
		for (unsigned int i = 0; i < allFramePoses.size(); i++)
		{
			glVertex3f((float)allFramePoses[i][0],
			           (float)allFramePoses[i][1],
			           (float)allFramePoses[i][2]);
		}
		glEnd();
	}
}

/**
 * @brief      { function_description }
 *
 * @param[in]  connectivity  The connectivity
 */
void PangolinDSOViewer::publishGraph(const std::map<long, Eigen::Vector2i> &connectivity)
{
	if (!setting_render_display3D) return;
	if (disableAllDisplay) return;

	model3DMutex.lock();
	connections.resize(connectivity.size() / 2);
	int runningID = 0;
	int totalActFwd = 0, totalActBwd = 0, totalMargFwd = 0, totalMargBwd = 0;
	for (std::pair<long, Eigen::Vector2i> p : connectivity)
	{
		int host = (int)(p.first >> 32);
		int target = (int)(p.first & (long)0xFFFFFFFF);

		assert(host >= 0 && target >= 0);
		if (host == target)
		{
			assert(p.second[0] == 0 && p.second[1] == 0);
			continue;
		}

		if (host > target) continue;

		connections[runningID].from = keyframesByKFID.count(host) == 0 ? 0 : keyframesByKFID[host];
		connections[runningID].to = keyframesByKFID.count(target) == 0 ? 0 : keyframesByKFID[target];
		connections[runningID].fwdAct = p.second[0];
		connections[runningID].fwdMarg = p.second[1];
		totalActFwd += p.second[0];
		totalMargFwd += p.second[1];

		long inverseKey = (((long)target) << 32) + ((long)host);
		Eigen::Vector2i st = connectivity.at(inverseKey);
		connections[runningID].bwdAct = st[0];
		connections[runningID].bwdMarg = st[1];

		totalActBwd += st[0];
		totalMargBwd += st[1];

		runningID++;
	}

	connections.resize(runningID);
	model3DMutex.unlock();
}


/**
 * @brief      { function_description }
 *
 * @param      frames     The frames
 * @param[in]  <unnamed>  { parameter_description }
 * @param      HCalib     The h calib
 */
void PangolinDSOViewer::publishKeyframes(
  std::vector<std::shared_ptr<FrameHessian>> &frames,
  bool final,
  CalibHessian* HCalib)
{
	if (!setting_render_display3D) return;
	if (disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(model3DMutex);
	for (std::shared_ptr<FrameHessian> fh : frames)
	{
		if (keyframesByKFID.find(fh->frameID) == keyframesByKFID.end())
		{
			KeyFrameDisplay* kfd = new KeyFrameDisplay();
			keyframesByKFID[fh->frameID] = kfd;
			keyframes.push_back(kfd);
		}
		keyframesByKFID[fh->frameID]->setFromKF(fh, HCalib);
	}
}

/**
 * @brief      { function_description }
 *
 * @param      frame   The frame
 * @param      HCalib  The h calib
 */
void PangolinDSOViewer::publishCamPose(FrameShell* frame,
                                       CalibHessian* HCalib)
{
	if (!setting_render_display3D) return;
	if (disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(model3DMutex);
	struct timeval time_now;
	gettimeofday(&time_now, NULL);
	lastNTrackingMs.push_back(((time_now.tv_sec - last_track.tv_sec) * 1000.0f + (time_now.tv_usec - last_track.tv_usec) / 1000.0f));
	if (lastNTrackingMs.size() > 10) lastNTrackingMs.pop_front();
	last_track = time_now;

	if (!setting_render_display3D) return;

	currentCam->setFromF(frame, HCalib);
	allFramePoses.push_back(frame->camToWorld.translation().cast<float>());
}

/**
 * @brief      Pushes a live frame.
 *
 * @param[in]  image  The image
 */
void PangolinDSOViewer::pushLiveFrame(std::shared_ptr<FrameHessian> image)
{
	if (!setting_render_displayVideo) return;
	if (disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(openImagesMutex);

	for (int i = 0; i < w * h; i++)
	{
		internalVideoImg->data[i][0] =
		  internalVideoImg->data[i][1] =
		    internalVideoImg->data[i][2] =
		      image->dI[i][0] * 0.8 > 255.0f ? 255.0 : image->dI[i][0] * 0.8;
		internalVideoImg_Right->data[i][0] = 255.0f;
		internalVideoImg_Right->data[i][1] = 255.0f;
		internalVideoImg_Right->data[i][2] = 255.0f;
	}

	videoImgChanged = true;
}

/**
 * @brief      Pushes a stereo live frame.
 *
 * @param[in]  image        The image
 * @param[in]  image_right  The image right
 */
void PangolinDSOViewer::pushStereoLiveFrame(std::shared_ptr<FrameHessian> image, std::shared_ptr<FrameHessian> image_right)
{
	if (!setting_render_displayVideo) return;
	if (disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(openImagesMutex);

	for (int i = 0; i < w * h; i++) {
		internalVideoImg->data[i][0] =
		  internalVideoImg->data[i][1] =
		    internalVideoImg->data[i][2] =
		      image->dI[i][0] * 0.8 > 255.0f ? 255.0 : image->dI[i][0] * 0.8;
		internalVideoImg_Right->data[i][0] =
		  internalVideoImg_Right->data[i][1] =
		    internalVideoImg_Right->data[i][2] =
		      image_right->dI[i][0] * 0.8 > 255.0f ? 255.0 : image_right->dI[i][0] * 0.8;
	}
	videoImgChanged = true;
}

/**
 * @brief      { function_description }
 *
 * @return     { description_of_the_return_value }
 */
bool PangolinDSOViewer::needPushDepthImage()
{
	return setting_render_displayDepth;
}

/**
 * @brief      Pushes a depth image.
 *
 * @param      image  The image
 */
void PangolinDSOViewer::pushDepthImage(MinimalImageB3* image)
{
	if (!setting_render_displayDepth) return;
	if (disableAllDisplay) return;

	boost::unique_lock<boost::mutex> lk(openImagesMutex);

	struct timeval time_now;
	gettimeofday(&time_now, NULL);
	lastNMappingMs.push_back(((time_now.tv_sec - last_map.tv_sec) * 1000.0f + (time_now.tv_usec - last_map.tv_usec) / 1000.0f));
	if (lastNMappingMs.size() > 10) lastNMappingMs.pop_front();
	last_map = time_now;

	memcpy(internalKFImg->data, image->data, w * h * 3);
	kfImgChanged = true;
}

}
}
