#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/version.hpp> 

using namespace cv;
using namespace std;

//该程序用来采集成对的双目图像。本文件共有两个main函数，分别用来采集双目标定板图像和双目图像（不进行交叉点检测）
//main函数1：这个main函数完成对标定板图像的采集
//正常进入采集流程后，按q键就会从左右相机中各拍摄一张图片，并会在当前图像上进行交叉点检测，如果均成功，就会把这对图像保存下来
//按escape键，退出采集过程
//main函数2：采集双目图像（不进行交叉点检测）

//此main函数为main1
//int main()
//{
//	//打开相机并设置好分辨率
//	int res_height = 720;
//	int res_width = 1280;
//	VideoCapture inputVideoLeft(0);
//	inputVideoLeft.set(cv::CAP_PROP_FRAME_HEIGHT, res_height);
//	inputVideoLeft.set(cv::CAP_PROP_FRAME_WIDTH, res_width);
//	if (!inputVideoLeft.isOpened())
//	{
//		cout << "打开左相机失败" << endl;
//		return -1;
//	}
//	else
//	{
//		cout << "左相机已经打开" << endl;
//	}
//
//	VideoCapture inputVideoRight(1);
//	inputVideoRight.set(cv::CAP_PROP_FRAME_HEIGHT, res_height);
//	inputVideoRight.set(cv::CAP_PROP_FRAME_WIDTH, res_width);
//	if (!inputVideoRight.isOpened())
//	{
//		cout << "打开右相机失败" << endl;
//		return -1;
//	}
//	else
//	{
//		cout << "右相机已经打开" << endl;
//	}
//
//	cv::Size iPatternSize(11, 8); //标定板上交叉点的个数
//	vector<Point2f> gCornersL, gCornersR; //分别存储检测自左右帧图像上的角点坐标
//	string imgname;
//	int frameIndex = 1;
//
//	Mat frameLeftOri, frameRightOri; //原始左右帧
//	Mat shrinkedFrameLeft, shrinkedFrameRight; //为了便于屏幕显示，我们把帧尺寸缩小一些
//	float shrinkScale = 0.5;
//	string imgDirName = "D:\\Books\\CV-Intro\\code\\chapter-17-stereo\\matlab-stereo\\stereo-imgs\\";
//
//	while (1)
//	{
//		inputVideoLeft >> frameLeftOri;
//		inputVideoRight >> frameRightOri;
//		if (frameLeftOri.empty() || frameRightOri.empty()) //两路视频有一路不能正常采集就进入下一次采集过程
//			continue;
//
//		//在屏幕上缩小回显当前两路视频
//		cv::resize(frameLeftOri, shrinkedFrameLeft, cv::Size(0, 0), shrinkScale, shrinkScale);
//		cv::resize(frameRightOri, shrinkedFrameRight, cv::Size(0, 0), shrinkScale, shrinkScale);
//		cv::imshow("Left Camera", shrinkedFrameLeft);
//		cv::imshow("Right Camera", shrinkedFrameRight);
//
//		char key = waitKey(1);
//		if (key == 27) //escape键的ASCII码是27，按Escape键就会退出采集，结束循环
//			break;
//		if (key == 'q' || key == 'Q') //按下q键进行图像采集 
//		{
//			//在当前左图像帧中进行交叉点检测
//			bool bPatternFound = findChessboardCorners(frameLeftOri, iPatternSize, gCornersL, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
//			if (!bPatternFound) //检测交叉点操作失败，说明当前左图像帧的图像质量不合格，继续等待采集下一帧
//			{
//				cout << "Can not find chessboard corners on the left image!\n";
//				continue;
//			}
//
//			bPatternFound = findChessboardCorners(frameRightOri, iPatternSize, gCornersR, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
//			if (!bPatternFound) //检测交叉点操作失败，说明当前右图像帧的图像质量不合格，继续等待采集下一帧
//			{
//				cout << "Can not find chessboard corners on the right image!\n";
//				continue;
//			}
//
//			//到达这里，说明左右图像都通过了检测交叉点测试
//			//显示交叉点检测结果，并将图像保存在本地磁盘
//			Mat iImageTemp = frameLeftOri.clone();
//			for (int j = 0; j < gCornersL.size(); j++)
//			{
//				//在当前帧上画出交叉点的坐标，这是为了可视化的目的
//				circle(iImageTemp, gCornersL[j], 10, Scalar(0, 0, 255), 2, 8, 0);
//			}
//			Mat iImageTempShrinked;
//			cv::resize(iImageTemp, iImageTempShrinked, cv::Size(0, 0), shrinkScale, shrinkScale);
//			cv::imshow("Left Camera", iImageTempShrinked);
//
//			iImageTemp = frameRightOri.clone();
//			for (int j = 0; j < gCornersR.size(); j++)
//			{
//				//在当前帧上画出交叉点的坐标，这是为了可视化的目的
//				circle(iImageTemp, gCornersR[j], 10, Scalar(0, 0, 255), 2, 8, 0);
//			}
//			cv::resize(iImageTemp, iImageTempShrinked, cv::Size(0, 0), shrinkScale, shrinkScale);
//			cv::imshow("Right Camera", iImageTempShrinked);
//
//			cv::waitKey(1000); //画有交叉点的帧，画面停留1秒
//			imgname = imgDirName + "left\\" + to_string(frameIndex) + ".jpg"; //左图像存储文件名称
//			imwrite(imgname, frameLeftOri);
//			imgname = imgDirName + "right\\" + to_string(frameIndex) + ".jpg"; //左图像存储文件名称
//			imwrite(imgname, frameRightOri);
//
//			frameIndex++;
//			cout << frameIndex - 1 << " images collected!" << endl;
//
//		}
//	}
//	cout << "完成标定板图像采集" << endl;
//	return 0;
//}


//此main函数为main2
//采集一对图像
int main()
{
	//打开相机并设置好分辨率
	int res_height = 720;
	int res_width = 1280;
	VideoCapture inputVideoLeft(0);
	inputVideoLeft.set(cv::CAP_PROP_FRAME_HEIGHT, res_height);
	inputVideoLeft.set(cv::CAP_PROP_FRAME_WIDTH, res_width);
	if (!inputVideoLeft.isOpened())
	{
		cout << "打开左相机失败" << endl;
		return -1;
	}
	else
	{
		cout << "左相机已经打开" << endl;
	}

	VideoCapture inputVideoRight(1);
	inputVideoRight.set(cv::CAP_PROP_FRAME_HEIGHT, res_height);
	inputVideoRight.set(cv::CAP_PROP_FRAME_WIDTH, res_width);
	if (!inputVideoRight.isOpened())
	{
		cout << "打开右相机失败" << endl;
		return -1;
	}
	else
	{
		cout << "右相机已经打开" << endl;
	}

	string imgname;
	int frameIndex = 1;

	Mat frameLeftOri, frameRightOri; //原始左右帧
	Mat shrinkedFrameLeft, shrinkedFrameRight; //为了便于屏幕显示，我们把帧尺寸缩小一些
	float shrinkScale = 0.5;
	string imgDirName = "D:\\Books\\CV-Intro\\code\\chapter-17-stereo\\matlab-stereo\\stereo-imgs\\";

	while (1)
	{
		inputVideoLeft >> frameLeftOri;
		inputVideoRight >> frameRightOri;
		if (frameLeftOri.empty() || frameRightOri.empty()) //两路视频有一路不能正常采集就进入下一次采集过程
			continue;

		//在屏幕上缩小回显当前两路视频
		cv::resize(frameLeftOri, shrinkedFrameLeft, cv::Size(0, 0), shrinkScale, shrinkScale);
		cv::resize(frameRightOri, shrinkedFrameRight, cv::Size(0, 0), shrinkScale, shrinkScale);
		cv::imshow("Left Camera", shrinkedFrameLeft);
		cv::imshow("Right Camera", shrinkedFrameRight);

		char key = waitKey(1);
		if (key == 27) //escape键的ASCII码是27，按Escape键就会退出采集，结束循环
			break;
		if (key == 'q' || key == 'Q') //按下q键进行图像采集 
		{
			cv::waitKey(1000); //画面停留1秒
			imgname = imgDirName + "test-left\\" + to_string(frameIndex) + ".jpg"; //左图像存储文件名称
			imwrite(imgname, frameLeftOri);
			imgname = imgDirName + "test-right\\" + to_string(frameIndex) + ".jpg"; //左图像存储文件名称
			imwrite(imgname, frameRightOri);

			frameIndex++;
			cout << frameIndex - 1 << " images collected!" << endl;

		}
	}
	cout << "完成标定板图像采集" << endl;
	return 0;
}
