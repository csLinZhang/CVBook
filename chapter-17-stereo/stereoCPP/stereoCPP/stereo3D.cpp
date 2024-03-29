// 双目立体重建
// 同济大学，张林，2024年4月

#include <opencv2/opencv.hpp>
#include <iostream>
#include "stereo.h"

//存放数据的路径
static string dataDir = "D:\\Books\\CV-Intro\\code\\chapter-17-stereo\\stereoCPP\\data\\";

//测试预先录制好的双目视频文件
int test_video_file() 
{
    CameraParam camera;//双目相机参数
	//读入两个相机的内参数
	cv::FileStorage fs(dataDir + "intrinsics.yml", cv::FileStorage::READ);
	fs["M1"] >> camera.cameraMatrixL;
	fs["D1"] >> camera.distCoeffL;
	fs["M2"] >> camera.cameraMatrixR;
	fs["D2"] >> camera.distCoeffR;

	//读入双目系统的外参数，即旋转矩阵和平移向量
	fs.open(dataDir + "extrinsics.yml", cv::FileStorage::READ);
	fs["R"] >> camera.R;
	fs["T"] >> camera.T;

	//图像的像素分辨率，这个值要根据所使用的相机型号情况设置
	camera.width = 1280;
	camera.height = 720;

    Stereo *detector = new Stereo(camera);
    int imageWidth = camera.width;      //单目图像的宽度
    int imageHeight = camera.height;    //单目图像的高度
	string left_video = dataDir + "test-left\\left-video.avi";
    string right_video = dataDir + "test-right\\right-video.avi";;
	cv::VideoCapture capL, capR;
    bool retL = get_video_capture(left_video, capL, imageWidth, imageHeight);
    bool retR = get_video_capture(right_video, capR, imageWidth, imageHeight);
	cv::Mat frameL, frameR;
    while (retL && retR) 
	{
        capL >> frameL;
        capR >> frameR;
        if (frameL.empty() or frameR.empty()) 
			break;
        detector->task(frameL, frameR, 20);
    }
    capL.release();         //释放对相机的控制
    capR.release();         //释放对相机的控制

    delete detector;
    return 0;

}

//测试双目摄像头输入
int test_camera() 
{
	CameraParam camera;//双目相机参数

	//读入两个相机的内参数
	cv::FileStorage fs(dataDir + "intrinsics.yml", cv::FileStorage::READ);
	fs["M1"] >> camera.cameraMatrixL;
	fs["D1"] >> camera.distCoeffL;
	fs["M2"] >> camera.cameraMatrixR;
	fs["D2"] >> camera.distCoeffR;

	//读入双目系统的外参数，即旋转矩阵和平移向量
	fs.open(dataDir + "extrinsics.yml", cv::FileStorage::READ);
	fs["R"] >> camera.R;
	fs["T"] >> camera.T;

	//图像的像素分辨率，这个值要根据所使用的相机型号情况设置
	camera.width = 1280;
	camera.height = 720;

    Stereo *detector = new Stereo(camera);
    int imageWidth = camera.width;       //单目图像的宽度
    int imageHeight = camera.height;     //单目图像的高度
    int camera1 = 0;                      //左摄像头ID号(请修改成自己左摄像头ID号)
    int camera2 = 1;                      //右摄像头ID号(请修改成自己右摄像头ID号)
	cv::VideoCapture capL, capR;
    bool retL = get_video_capture(camera1, capL, imageWidth, imageHeight);
    bool retR = get_video_capture(camera2, capR, imageWidth, imageHeight);
	cv::Mat frameL, frameR;
    while (retL && retR) 
	{
        capL >> frameL;
        capR >> frameR;
        if (frameL.empty() or frameR.empty()) break;
        detector->task(frameL, frameR, 20);
    }
    capL.release();         //释放对相机的控制
    capR.release();         //释放对相机的控制
    delete detector;
    return 0;
}

//从一对双目图像中重建3D场景
int test_pair_image_file() 
{
    CameraParam camera;//双目相机参数

	//读入两个相机的内参数
	cv::FileStorage fs(dataDir + "intrinsics.yml", cv::FileStorage::READ);
	fs["M1"] >> camera.cameraMatrixL;
	fs["D1"] >> camera.distCoeffL;
	fs["M2"] >> camera.cameraMatrixR;
	fs["D2"] >> camera.distCoeffR;
	
	//读入双目系统的外参数，即旋转矩阵和平移向量
	fs.open(dataDir + "extrinsics.yml", cv::FileStorage::READ);
	fs["R"] >> camera.R;
	fs["T"] >> camera.T;
	
	//图像的像素分辨率，这个值要根据所使用的相机型号情况设置
	camera.width = 1280;
	camera.height = 720;

    Stereo *detector = new Stereo(camera);
	//从预先拍摄的一对双目图像中恢复3D场景
	cv::Mat frameL = cv::imread(dataDir + "test-left\\2.jpg", cv::IMREAD_COLOR);
	cv::Mat frameR = cv::imread(dataDir + "test-right\\2.jpg", cv::IMREAD_COLOR);
    detector->task(frameL, frameR, 0);
	detector->generatePointCloud();
    delete detector;
    return 0;
}

int main() 
{
    //测试一对左右图像
    test_pair_image_file();
    //测试demo视频文件
    //test_video_file();
    //测试双目摄像头(双USB连接线的双目摄像头)
    //test_camera();
    return 0;
}
