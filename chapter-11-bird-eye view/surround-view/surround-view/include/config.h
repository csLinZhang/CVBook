#ifndef CONFIG_H_
#define CONFIG_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
using namespace std;

//鸟瞰环视图的分辨率
const int BIRDS_EYE_WIDTH = 600;
const int BIRDS_EYE_HEIGHT = 600;
//环视图中，中心黑块的尺寸，由于中心是车身所在位置，所以这部分是没有图像内容的
const int CENTER_HEIGHT = 300;
const int CENTER_WIDTH = 150;

const cv::Size iCameraImageSize(1280 , 1080);

//演示数据文件夹
const string dataDir = "D:\\Books\\CV-Intro\\code\\chapter-11-bird-eye view\\surround-view\\data\\";

#endif