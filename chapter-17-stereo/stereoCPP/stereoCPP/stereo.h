
#ifndef STEREO_H
#define STEREO_H

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

//用于存放左目校正化图像上每个像素点在左目校正相机坐标系下的三维坐标
static cv::Mat xyz_coord;  

//双目视觉系统参数
struct CameraParam 
{
    int width;           //物理图像的宽度
    int height;          //物理图像的高度
	cv::Mat cameraMatrixL;   //物理左目相机内参(3×3)
	cv::Mat distCoeffL;      //物理左目相机畸变系数(5×1)
	cv::Mat cameraMatrixR;   //物理右目相机内参(3×3)
	cv::Mat distCoeffR;      //物理右目相机畸变系数(5×1)
	cv::Mat T;               //双目系统外参中的平移部分(3×1)
	cv::Mat R;               //双目系统外参中的旋转矩阵(3×3)
};

/***
 * 显示图像
 * @param winname 窗口名称
 * @param image 图像
 * @param delay 显示延迟，0表示阻塞显示
 * @param flags 显示方式
 */
static void show_image(const string &winname, cv::Mat &image, int delay = 0) 
{
    cv::namedWindow(winname, cv::WINDOW_AUTOSIZE);
    cv::imshow(winname, image);
    cv::waitKey(delay);
}

/***
 * 读取视频文件
 * @param video_file 视频文件
 * @param cap 视频流对象
 * @param width 设置图像的宽度
 * @param height 设置图像的高度
 * @param fps 设置视频播放频率
 * @return
 */
static bool get_video_capture(string video_file, cv::VideoCapture &cap, int width = -1, int height = -1, int fps = -1) 
{
    //VideoCapture video_cap;
    cap.open(video_file);
    if (width > 0 && height > 0) 
	{
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width); //设置图像的宽度
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height); //设置图像的高度
    }
    if (fps > 0) 
	{
        cap.set(cv::CAP_PROP_FPS, fps);
    }
    if (!cap.isOpened())//判断是否读取成功
    {
        return false;
    }
    return true;
}

/***
 * 读取摄像头
 * @param camera_id 摄像头ID号，默认从0开始
 * @param cap 视频流对象
 * @param width 设置图像的宽度
 * @param height 设置图像的高度
 * @param fps 设置视频播放频率
 * @return
 */
static bool get_video_capture(int camera_id, cv::VideoCapture &cap, int width = -1, int height = -1, int fps = -1) 
{
    //VideoCapture video_cap;
    cap.open(camera_id);    //摄像头ID号，默认从0开始
    if (width > 0 && height > 0) {
        cap.set(cv::CAP_PROP_FRAME_WIDTH, width); //设置捕获图像的宽度
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, height);  //设置捕获图像的高度
    }
    if (fps > 0) {
        cap.set(cv::CAP_PROP_FPS, fps);
    }
    if (!cap.isOpened()) //判断是否成功打开相机
    {
        return false;
    }
    return true;
}

class Stereo 
{
public:
    /***
     * 构造函数，初始化Stereo
     * camera，双目相机参数结构体
     */
    Stereo(CameraParam camera);
    ~Stereo();
    void task(cv::Mat frameL, cv::Mat frameR, int delay = 0);
    void get_rectify_image(cv::Mat &imgL, cv::Mat &imgR, cv::Mat &rectifiedL, cv::Mat &rectifiedR);
    void get_disparity(cv::Mat &dispL);
    void get_3dpoints(cv::Mat &disp, cv::Mat &points_3d);
    //将输入深度图转换为伪彩色图，方便可视化
    void get_visual_depth(cv::Mat &depth, cv::Mat &colormap, float clip_max = 6000.0);
    void show_rectify_result(cv::Mat rectifiedL, cv::Mat rectifiedR);
    void show_2dimage(cv::Mat &points_3d, cv::Mat &disp, int delay);
	void generatePointCloud();
    void clip(cv::Mat &src, float vmin, float vmax);

public:
	cv::Size image_size;                                  // 图像宽高(width,height)
	cv::Rect validROIL;                                   // 图像校正之后，会对图像进行裁剪，这里的左视图裁剪之后的区域
	cv::Rect validROIR;                                   // 图像校正之后，会对图像进行裁剪，这里的右视图裁剪之后的区域
	cv::Mat mapLx, mapLy, mapRx, mapRy;                   // 映射表
	cv::Mat Rl, Rr, Pl, Pr, Q;                            // 校正后的旋转矩阵R，投影矩阵P, 重投影矩阵Q
	cv::Mat dispL;                                        // 视差图(CV_32F)
	cv::Mat disp_colormap;                                // 视差图可视化图(CV_8UC3)
	cv::Mat depth;                                        // 深度图(CV_32F)
	cv::Mat depth_colormap;                               // 深度图可视化图(CV_8UC3)
	cv::Mat points_3d;                                    // 世界坐标图(CV_32F)
	cv::Ptr<cv::StereoSGBM> sgbm;
	cv::Mat rectifiedL, rectifiedR;
	CameraParam camera_params;
};

#endif //STEREO_H
