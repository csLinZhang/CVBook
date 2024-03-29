#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <io.h>

using namespace cv;
using namespace std;

//该程序用来采集成对的双目图像以及视频。本文件共有三个main函数，分别用来采集双目标定板图像、双目图像（不进行交叉点检测）和双目视频
//main函数1：这个main函数完成对标定板图像的采集
//正常进入采集流程后，按q键就会从左右相机中各拍摄一张图片，并会在当前图像上进行交叉点检测，如果均成功，就会把这对图像保存下来
//按escape键，退出采集过程
//main函数2：采集双目图像（不进行交叉点检测）
//main函数3：采集双目视频

//存放数据的路径
static string dataDir = "D:\\Books\\CV-Intro\\code\\chapter-17-stereo\\stereoCPP\\data\\";

//假设采集的标定板左右目图像分别存放在了path\left和path\right文件夹中
//该函数得到“path\left”的文件夹下所有类型为“ext”的文件路径的列表，并将它们以及同名的“path\right”路径下的文件路径填入filelist中
//这样，filelist一定有偶数行，每两行分别是对应的左右目图像的文件路径
void get_need_file(string path, vector<string>& filelist, string ext)
{
	intptr_t file_handle = 0;
	struct _finddata_t file_info;
	string temp;
	if ((file_handle = _findfirst(temp.assign(path + "left").append("/*" + ext).c_str(), &file_info)) != -1)
	{
		do
		{
			filelist.push_back(temp.assign(path + "left").append("\\").append(file_info.name));
			filelist.push_back(temp.assign(path + "right").append("\\").append(file_info.name));
		} 
		while (_findnext(file_handle, &file_info) == 0);
		_findclose(file_handle);
	}
}

//main函数1：这个main函数完成对标定板图像的采集
//正常进入采集流程后，按q键就会从左右相机中各拍摄一张图片，并会在当前图像上进行交叉点检测，如果均成功，就会把这对图像保存下来
//按escape键，退出采集过程
//int main(int argc, char* argv[])
//{
//	//打开相机并设置好分辨率，分辨率需要根据读者自己的情况进行调整
//	int res_Width = 1280;
//	int res_Height = 720;
//	VideoCapture inputVideoLeft(0);
//	inputVideoLeft.set(cv::CAP_PROP_FRAME_HEIGHT, res_Height);
//	inputVideoLeft.set(cv::CAP_PROP_FRAME_WIDTH, res_Width);
//	if (!inputVideoLeft.isOpened())
//	{
//		cout << "Fail to start the left camera" << endl;
//		return -1;
//	}
//	else
//	{
//		cout << "The left camera is ready" << endl;
//	}
//
//	VideoCapture inputVideoRight(1);
//	inputVideoRight.set(cv::CAP_PROP_FRAME_HEIGHT, res_Height);
//	inputVideoRight.set(cv::CAP_PROP_FRAME_WIDTH, res_Width);
//	if (!inputVideoRight.isOpened())
//	{
//		cout << "Fail to start the right camera" << endl;
//		return -1;
//	}
//	else
//	{
//		cout << "The right camera is ready" << endl;
//	}
//
//	cv::Size iPatternSize(11, 8); //标定板上交叉点的个数
//	vector<Point2f> gCornersL, gCornersR; //分别存储检测自左右帧图像上的角点坐标
//	string imgname;
//	int frameIndex = 1;
//
//	Mat frameLeftOri, frameRightOri; //原始左右帧
//	Mat shrinkedFrameLeft, shrinkedFrameRight; //为了便于屏幕显示，我们把帧尺寸缩小一些
//	
//	float shrinkScale = 0.6; //为便于查看，缩小显示
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
//			//到达这里，说明左右图像都通过了检测交叉点测试，将图像保存在本地磁盘
//
//			Mat iImageTemp = frameLeftOri.clone();
//			for (int j = 0; j < gCornersL.size(); j++)
//			{
//				//在当前帧上画出交叉点的坐标，这是为了可视化的目的
//				circle(iImageTemp, gCornersL[j], 10, Scalar(0, 0, 255), 2, 8, 0);
//			}
//
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
//			imgname = dataDir + "left\\" + to_string(frameIndex) + ".jpg"; //左图像存储文件名称
//			imwrite(imgname, frameLeftOri);
//			imgname = dataDir + "right\\" + to_string(frameIndex) + ".jpg"; //左图像存储文件名称
//			imwrite(imgname, frameRightOri);
//
//			frameIndex++;
//			cout << frameIndex - 1 << " images collected!" << endl;
//
//		}
//	}
//	cout << "Finish collecting calibration board images." << endl;
//
//	//接下来要将双目图像的文件路径写入一个本地文本文件，供下一步双目标定操作使用
//    //string file_path = R"(D:\Books\CV-Intro\code\chapter-17-3DReconstruction\stereocalib\imgs\)";
//    vector<string> stereoImgsFileList;
//    string need_extension = ".jpg";
//    get_need_file(dataDir, stereoImgsFileList, need_extension);
//    for (int i = 0; i < stereoImgsFileList.size(); i++)
//    {
//        cout << "File " << i + 1 << " is:" << endl;
//        cout << stereoImgsFileList[i] << endl;
//    }
//    cout << endl << "Find " << stereoImgsFileList.size() << " file(s)." << endl;
//
//    //将stereoImgsFileList中的文件路径写入本地文本文件
//    ofstream txtfile(dataDir + "\\imgpaths.txt",ios::out);
//    for (int index = 0; index < stereoImgsFileList.size(); index++)
//    {
//        txtfile << stereoImgsFileList[index] << endl;
//    }
//    txtfile.close();
//
//	//读取并显示一下以上存放双目图像路径的内容
//    ifstream storedtxtfile(dataDir + "\\imgpaths.txt",ios::in);
//    string currentfilepath;
//    while(getline(storedtxtfile, currentfilepath))
//    {
//        cout << "This image path is:" << endl;
//        cout << currentfilepath << endl;
//    }
//    storedtxtfile.close();
//	return 0;
//}

//此main函数为main2
//采集成对双目图像，存储在本地dataDir\test-left和dataDir\test-right目录下
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
//		cout << "Fail to start the left camera." << endl;
//		return -1;
//	}
//	else
//	{
//		cout << "The left camera is ready." << endl;
//	}
//
//	VideoCapture inputVideoRight(1);
//	inputVideoRight.set(cv::CAP_PROP_FRAME_HEIGHT, res_height);
//	inputVideoRight.set(cv::CAP_PROP_FRAME_WIDTH, res_width);
//	if (!inputVideoRight.isOpened())
//	{
//		cout << "Fail to start the right camera." << endl;
//		return -1;
//	}
//	else
//	{
//		cout << "The right camera is ready." << endl;
//	}
//
//	string imgname;
//	int frameIndex = 1;
//
//	Mat frameLeftOri, frameRightOri; //原始左右帧
//	Mat shrinkedFrameLeft, shrinkedFrameRight; //为了便于屏幕显示，我们把帧尺寸缩小一些
//	float shrinkScale = 0.5;
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
//			cv::waitKey(1000); //画面停留1秒
//			imgname = dataDir + "test-left\\" + to_string(frameIndex) + ".jpg"; //左图像存储文件名称
//			imwrite(imgname, frameLeftOri);
//			imgname = dataDir + "test-right\\" + to_string(frameIndex) + ".jpg"; //左图像存储文件名称
//			imwrite(imgname, frameRightOri);
//
//			frameIndex++;
//			cout << frameIndex - 1 << " images collected!" << endl;
//
//		}
//	}
//	cout << "Image collection is finished." << endl;
//	return 0;
//}

//此main函数为main3
//采集双目视频，左目视频存储在本地dataDir\test-left目录下，右目视频存储在本地dataDir\test-right目录下
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
		cout << "Fail to start the left camera." << endl;
		return -1;
	}
	else
	{
		cout << "The left camera is ready." << endl;
	}

	VideoCapture inputVideoRight(1);
	inputVideoRight.set(cv::CAP_PROP_FRAME_HEIGHT, res_height);
	inputVideoRight.set(cv::CAP_PROP_FRAME_WIDTH, res_width);
	if (!inputVideoRight.isOpened())
	{
		cout << "Fail to start the right camera." << endl;
		return -1;
	}
	else
	{
		cout << "The right camera is ready." << endl;
	}

	string imgname;
	int frameIndex = 1;

	Mat frameLeftOri, frameRightOri; //原始左右帧
	Mat shrinkedFrameLeft, shrinkedFrameRight; //为了便于屏幕显示，我们把帧尺寸缩小一些
	float shrinkScale = 0.5;

	cv::Size sWH(res_width, res_height);
	//确定好输出路径，将采集的两路视频保存在本地
	cv::VideoWriter outputVideoLeft;
	outputVideoLeft.open(dataDir+ "test-left\\left-video.avi", cv::VideoWriter::fourcc('M', 'P', '4', '2'), 25.0, sWH);
	cv::VideoWriter outputVideoRight;
	outputVideoRight.open(dataDir + "test-right\\right-video.avi", cv::VideoWriter::fourcc('M', 'P', '4', '2'), 25.0, sWH);
	
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
		outputVideoLeft << frameLeftOri;
		outputVideoRight << frameRightOri;
		frameIndex++;
		cout << frameIndex - 1 << " frames collected!" << endl;

		char key = waitKey(1);
		if (key == 27) //escape键的ASCII码是27，按Escape键就会退出采集，结束循环
		{
			break;
		}
	}

	outputVideoLeft.release();
	outputVideoRight.release();
	cout << "Video collection is finished." << endl;
	return 0;
}