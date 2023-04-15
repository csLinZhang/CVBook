#ifndef CONFIG_H_
#define CONFIG_H_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <iostream>
using namespace std;

//�����ͼ�ķֱ���
const int BIRDS_EYE_WIDTH = 600;
const int BIRDS_EYE_HEIGHT = 600;
//����ͼ�У����ĺڿ�ĳߴ磬���������ǳ�������λ�ã������ⲿ����û��ͼ�����ݵ�
const int CENTER_HEIGHT = 300;
const int CENTER_WIDTH = 150;

const cv::Size iCameraImageSize(1280 , 1080);

//��ʾ�����ļ���
const string dataDir = "D:\\Self Made Books\\CV-Intro\\code\\chapter-11-bird-eye view\\01-surround-view\\surround-view\\data\\";

#endif