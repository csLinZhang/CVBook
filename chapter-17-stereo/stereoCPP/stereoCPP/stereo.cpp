//此文件为双目视觉系统三维重建主文件
//同济大学，张林，2024年5月

#include "stereo.h"
#include "opencv2/ximgproc.hpp"
#include <Eigen/Core>
#include <pangolin/pangolin.h>
using namespace cv;

typedef Eigen::Matrix<double, 6, 1> Vector6d; //点云中一个点的数据结构，6维向量

Stereo::Stereo(CameraParam camera) 
{
	this->camera_params = camera;
    this->image_size = Size(camera.width, camera.height);

    //双目校正
    stereoRectify(camera.cameraMatrixL, camera.distCoeffL,
                  camera.cameraMatrixR, camera.distCoeffR,
                  image_size, camera.R,
                  camera.T, Rl, Rr, Pl, Pr, Q,
                  CALIB_ZERO_DISPARITY,
                  0, image_size,
                  &validROIL, &validROIR);
	cout << Q;
	//计算从校正化图像到原始物理图像的位置映射表
    initUndistortRectifyMap(camera.cameraMatrixL, camera.distCoeffL, Rl, Pl, image_size, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(camera.cameraMatrixR, camera.distCoeffR, Rr, Pr, image_size, CV_32FC1, mapRx, mapRy);

    //SGBM算法初始化
    int mindisparity = 0;                                                   
    int blockSize = 9; //比对块的大小
    int numDisparities = 6 * 16; //最大的视差，要被16整除
    int P1 = 8 * 3 * blockSize;  //惩罚系数1
    int P2 = 32 * 3 * blockSize; //惩罚系数2
    sgbm = cv::StereoSGBM::create(mindisparity, numDisparities, blockSize);
    sgbm->setP1(P1);
    sgbm->setP2(P2);
    sgbm->setDisp12MaxDiff(1);                                             
    sgbm->setUniquenessRatio(10);                                        
    sgbm->setSpeckleWindowSize(50);                                     
    sgbm->setSpeckleRange(32);                                              
    sgbm->setPreFilterCap(63);                                               
    sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
}

Stereo::~Stereo() 
{
    mapLx.release();
    mapLy.release();
    mapRx.release();
    mapRy.release();
    Rl.release();
    Rr.release();
    Pl.release();
    Pr.release();
    Q.release();
    sgbm.release();
}

void Stereo::task(Mat frameL, Mat frameR, int delay) 
{
    //根据双目系统参数，完成图像的校正
    this->get_rectify_image(frameL, frameR, rectifiedL, rectifiedR);
    // 绘制等间距平行线，检查立体校正的效果
	this->show_rectify_result(rectifiedL, rectifiedR); 
    //获得视差图
    this->get_disparity(this->dispL);
    //用于存放每个像素点距离相机镜头的三维坐标
    this->get_3dpoints(this->dispL, this->points_3d);
    xyz_coord = this->points_3d;
    // 显示视差图效果
    this->show_2dimage(this->points_3d, this->dispL, delay);
}

void Stereo::get_rectify_image(Mat &imgL, Mat &imgR, Mat &rectifiedL, Mat &rectifiedR)
{
    //经过remap之后，左右相机的图像已经共面并且行对准
    remap(imgL, rectifiedL, mapLx, mapLy, INTER_LINEAR);
    remap(imgR, rectifiedR, mapRx, mapRy, INTER_LINEAR);
}

void Stereo::show_rectify_result(cv::Mat rectifiedL, cv::Mat rectifiedR) 
{
    //显示在同一张图上
    Mat canvas;
    double sf;
    int w, h;
    sf = 600. / MAX(image_size.width, image_size.height);
    w = cvRound(image_size.width * sf);
    h = cvRound(image_size.height * sf);
    canvas.create(h, w * 2, CV_8UC3);

    //左图像画到画布上
    Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分
    resize(rectifiedL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);    //把图像缩放到跟canvasPart一样大小
 
    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));                                        //获得画布的另一部分
    resize(rectifiedR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);

    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
    show_image("rectified", canvas, 1);
}

//根据左右目校正化图像计算视差图
void Stereo::get_disparity(Mat &dispL) 
{
	cv::Mat imgL = this->rectifiedL.clone();
	cv::Mat imgR = this->rectifiedR.clone();
    cvtColor(imgL, imgL, COLOR_BGR2GRAY);
    cvtColor(imgR, imgR, COLOR_BGR2GRAY);
	//跟怒semi-global block matching算法计算视差图
    sgbm->compute(imgL, imgR, dispL);

	//可视化视差
	//Mat GT_disp_vis;
	//cv::ximgproc::getDisparityVis(dispL, GT_disp_vis, 1.0);
	//cv::imshow("disparity before wls", GT_disp_vis);

	//string imgname = "./disparity_without_wls.jpg"; //左图像存储文件名称
	//imwrite(imgname, GT_disp_vis);
	  
    //对初始得到的视差图进行wls滤波处理
    int lmbda = 80000;
    float sigma = 1.3f;
    Ptr<StereoMatcher> matcherR = cv::ximgproc::createRightMatcher(sgbm);
    cv::Mat dispR;
    matcherR->compute(imgR, imgL, dispR);
    auto filter = cv::ximgproc::createDisparityWLSFilter(sgbm);
    filter->setLambda(lmbda);
    filter->setSigmaColor(sigma);
    filter->filter(dispL, imgL, dispL, dispR);
	//此时的dispL是经过wls滤波处理之后的结果
	//cv::ximgproc::getDisparityVis(dispL, GT_disp_vis, 1.0);
	//cv::imshow("disparity after wls", GT_disp_vis);

	//imgname = "./disparity_wls.jpg"; //左图像存储文件名称
	//imwrite(imgname, GT_disp_vis);
    //除以16得到真实视差（因为SGBM算法得到的视差是×16的）
    dispL.convertTo(dispL, CV_32F, 1.0 / 16);
}

void Stereo::get_3dpoints(Mat &disp, Mat &points_3d) 
{
    //由视差图以及矩阵Q，计算出与左校正化图像对应的三维点云
	//矩阵Q就是教材中17-38所示的矩阵Q
    reprojectImageTo3D(disp, points_3d, Q, true);
    //points_3d = points_3d ;
}

//利用Pangolin可视化库，对点云pointcloud进行可视化
void showPointCloud(const vector<Vector6d, Eigen::aligned_allocator<Vector6d>> &pointcloud) 
{
	if (pointcloud.empty()) 
	{
		cerr << "Point cloud is empty!" << endl;
		return;
	}

	pangolin::CreateWindowAndBind("Point Cloud Viewer", 1024, 768);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	pangolin::OpenGlRenderState s_cam(
		pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
		pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
	);

	pangolin::View &d_cam = pangolin::CreateDisplay()
		.SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
		.SetHandler(new pangolin::Handler3D(s_cam));

	while (pangolin::ShouldQuit() == false) 
	{
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		d_cam.Activate(s_cam);
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

		glPointSize(2);
		glBegin(GL_POINTS);
		for (auto &p : pointcloud) 
		{
			glColor3d(p[3] / 255.0, p[4] / 255.0, p[5] / 255.0);
			glVertex3d(p[0], p[1], p[2]);
		}
		glEnd();
		pangolin::FinishFrame();
	}
	return;
}

//生成并显示3D点云
void Stereo::generatePointCloud()
{
	//系统的物理度量单位都是毫米，显示点云的时候单位是米，因此整体坐标要除以1000
	double depthScale = 1000.0; 
	vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
	pointcloud.reserve(1000000);
	//彩色点云，左目校正化图像每个像素对应一个三维点，该三维点的颜色就是该像素颜色
	//这样每个点云点的数据就是个6维向量，(x,y,z,b,g,r)
	cv::Mat color = this->rectifiedL;
	for (int v = 0; v < color.rows; v++)
	{
		for (int u = 0; u < color.cols; u++)
		{
			unsigned int d = depth.ptr<unsigned short>(v)[u]; // 深度值
			if (d == 0) continue; // 为0表示没有测量到
			Eigen::Vector3d point;
			//填充三维坐标信息
			point[0] = this->points_3d.at<cv::Vec3f>(v, u)[0] / depthScale;
			point[1] = this->points_3d.at<cv::Vec3f>(v, u)[1] / depthScale;
			point[2] = this->points_3d.at<cv::Vec3f>(v, u)[2] / depthScale;

			Vector6d p;
			p.head<3>() = point;
			//填充彩色信息
			p[5] = color.data[v * color.step + u * color.channels()];   // blue
			p[4] = color.data[v * color.step + u * color.channels() + 1]; // green
			p[3] = color.data[v * color.step + u * color.channels() + 2]; // red
			pointcloud.push_back(p);
		}
	}
	cout << "The point cloud has " << pointcloud.size() << " points." << endl;
	//把点云可视化出来
	showPointCloud(pointcloud);
}

void Stereo::show_2dimage(Mat &points_3d, Mat &disp, int delay) 
{
    //显示结果
    vector<Mat> xy_depth;
    split(points_3d, xy_depth); //分离出深度通道
    depth = xy_depth[2];
    get_visual_depth(disp, disp_colormap);
    //show_image("left", this->rectifiedL, 1);
    //show_image("right", this->rectifiedR, 1);
    //show_image("disparity-color", disp_colormap, 1);
    // 可视化深度图
    get_visual_depth(depth, depth_colormap);
    show_image("depth-color", depth_colormap, delay);
}

//用热力图来显示深度
void Stereo::get_visual_depth(cv::Mat &depth, cv::Mat &colormap, float clip_max)                   
{
    clip(depth, 0.0, clip_max);
    Mat int8disp = Mat(depth.rows, depth.cols, CV_8UC1);                       
    normalize(depth, int8disp, 0, 255, NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(int8disp, colormap, cv::COLORMAP_JET);
}

//深度减裁，超过vmax的深度值被置为vmax，小于vmin的深度值被置为vmin
void Stereo::clip(cv::Mat &src, float vmin, float vmax) 
{
    int h = src.rows;
    int w = src.cols;
    if (src.isContinuous() && src.isContinuous()) 
	{
        h = 1;
        w = w * src.rows * src.channels();
    }
    for (int i = 0; i < h; i++) 
	{
        float *sptr = src.ptr<float>(i);
        for (int j = 0; j < w; j++) 
		{
            sptr[j] = sptr[j] < vmax ? sptr[j] : vmax;
            sptr[j] = sptr[j] > vmin ? sptr[j] : vmin;
        }
    }
}