//同济大学，张林，2024年4月
//此程序示范了如何用opencv来进行鱼眼相机内参标定，以及如何利用获取的相机内参来进行鱼眼图像的畸变去除。
//该程序有3个main函数，编译时，每次都需要注释掉其他两个，只保留一个main函数
//main函数1：完成对标定板图像的采集
//main函数2：基于main函数1所采集的标定板图像，完成相机内参标定，并将参数文件存储在本地磁盘
//main函数3：从由main函数2所得到的相机内参文件中，读入相机内参，对实时输入视频进行实时去畸变

#include <string>
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
#include <fstream>

using namespace cv;
using namespace std;

//存储数据的文件夹路径
//data/imgs，存放采集的标定板图像
//data下面存放相机内参文件
string dataDir = "D:\\Books\\CV-Intro\\code\\chapter-10-imaging model and intrinsics calibration\\fisheyeCameraCalib\\data";

//main函数1
//这个main函数完成对标定板图像的采集
//正常进入采集流程后，按q键就会拍摄一张图像并会在当前图像上进行交叉点检测，
//如果成功，就会把这张图像保存下来
//按escape键，退出采集过程
//int main()
//{
//	cout << "开始采集标定板图像" << endl;
//	//imgDir，存放所采集的标定板图像的文件夹路径
//	string imgDir = dataDir + "\\imgs\\";
//	Mat frame;
//	string imgname;
//	int frameIndex = 1;
//	//交叉点的个数，如果是10*7块的标定板，其有效交叉点是9*6个
//	cv::Size iPatternSize(9, 6); 
//	vector<Point2f> gCorners;
//
//	//打开相机并设置相机分辨率，相机的序号从0开始
//	VideoCapture inputVideo(0); 
//	inputVideo.set(cv::CAP_PROP_FRAME_HEIGHT, 768);
//	inputVideo.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
//
//	if (!inputVideo.isOpened()) 
//	{
//		cout << "打开相机失败 " << endl;
//		return -1;
//	}
//	else 
//	{
//		cout << "相机已打开" << endl;
//	}
//
//	int imgsCollected = 0;
//	while (1) 
//	{
//		inputVideo >> frame;
//		if (frame.empty()) 
//			continue;
//		imshow("Camera", frame);
//		char key = waitKey(1);
//		if (key == 27) //escape键的ASCII码是27，按Escape键就会退出采集，结束循环
//			break;
//		if (key == 'q' || key == 'Q') 
//		{
//			//在当前图像帧中进行交叉点检测
//			bool bPatternFound = findChessboardCorners(frame, iPatternSize, gCorners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE +
//				CALIB_CB_FAST_CHECK);
//			if (!bPatternFound) //检测交叉点操作失败，说明当前帧的图像质量不合格，继续等待采集下一帧
//			{
//				cout << "Can not find chessboard corners!\n";
//				continue;
//			}
//			else //将当前帧frame存储在本地磁盘路径imgDirName中
//			{
//				Mat iImageTemp = frame.clone();
//				for (int j = 0; j < gCorners.size(); j++)
//				{
//					//在当前帧上画出交叉点的坐标，这是为了可视化的目的
//					circle(iImageTemp, gCorners[j], 10, Scalar(0, 0, 255), 2, 8, 0);
//				}
//				cv::imshow("Camera", iImageTemp);
//				cv::waitKey(1000); //画有交叉点的帧，画面停留1秒
//				imgname = imgDir + to_string(frameIndex++) + ".jpg"; //为当前帧起好名字，存储
//				imwrite(imgname, frame);
//				imgsCollected++;
//				cout << imgsCollected << " images collected!" << endl;
//			}
//		}
//	}
//	cout << "完成标定板图像采集" << endl;
//	return 0;
//}

//main函数2
//该main函数完成从一组标定板图像中对鱼眼相机内参进行标定的任务
//相机畸变模型采用鱼眼相机模型，适用于广角鱼眼相机去畸变任务
int main(int argc, char* argv[])
{
	cv::Size iPatternSize(9, 6); //标定板上的交叉点维度，10*7格的标定板，其交叉点为9*6
	string imgDir = dataDir + "\\imgs\\";
	//存储相机内参标定结果的文件
	FileStorage camParamsFile(dataDir+"\\camParams.xml", FileStorage::WRITE);

	//3*3相机内参矩阵
	cv::Mat mIntrinsicMatrix;
	//畸变系数
	cv::Mat mDistortion;
	//标定板外参，每个外参由一个轴角向量与一个平移向量组成
	std::vector<cv::Vec3d> gRotationVectors;
	std::vector<cv::Vec3d> gTranslationVectors;
	vector<cv::String> gFileNames; //存储所有标定板图像文件全名称
	cv::glob(imgDir, gFileNames); //得到所有标定板图像文件路径
	cout << "Load images" << endl;
	int nImageCount = gFileNames.size(); //标定板图像的数目
	vector<Point2f> gCorners; //存储一张图像上交叉点坐标信息.
	//gAllCorners，存储所有标定板图像上的交叉点信息，这是个vector的vector,
	//每个vector是一个gCorners
	vector<vector<Point2f>>  gAllCorners;
	//gImages,存储所有标定板图像，是个元素类型为Mat的向量，显然每个向量元素是个图像矩阵
	vector<Mat>  gImages;
	//存储合法图像文件名称，合法图像指的是能被正确检测到交叉点的图像
	vector<string> strValidImgFileNames; 
	//对于每一张采集到的标定板图像
	for (int i = 0; i < nImageCount; i++)
	{
		//读入该图像
		string aImageFileName = gFileNames[i];
		cv::Mat iImage = imread(aImageFileName);
		cout << "Filename " << aImageFileName << endl;
		//转换为灰度图像，因为亚像素级别的交叉点精准检测是在灰度图像上进行的
		Mat iImageGray;
		cvtColor(iImage, iImageGray, cv::COLOR_BGR2GRAY);
		bool bPatternFound = findChessboardCorners(iImage, iPatternSize, gCorners, 
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		if (!bPatternFound) //这张图像的交叉点检测失败
		{
			cout << "Can not find chessboard corners in " << aImageFileName << "\n";
			continue;
		}
		else
		{
			//对初始检测的交叉点坐标进行进一步精化
			cornerSubPix(iImageGray, gCorners, Size(11, 11), Size(-1, -1), 
				TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
			
			//为了可视化，把当前图像的交叉点检测结果画在图像上，显示出来，停留1000
			Mat iImageTemp = iImage.clone();
			for (int j = 0; j < gCorners.size(); j++)
			{
				//在图像上画出交叉点
				circle(iImageTemp, gCorners[j], 10, Scalar(0, 0, 255), 2, 8, 0);
			}
			cv::imshow("corners", iImageTemp);
			cv::waitKey(1000);

			//把从当前图像上检测到的交叉点放入gAllCorners，相应的图像放入gImages
			gAllCorners.push_back(gCorners);
			gImages.push_back(iImage);
			strValidImgFileNames.push_back(aImageFileName);
		}
	}
	
	//nImageCount，能正确进行交叉点检测的图像的个数
	nImageCount = gImages.size();
	cout << "能成功进行交叉点检测的图像的个数为" << nImageCount << endl;
	
	//生成标定板交叉点的三维坐标gObjectPoints，
	//每张图像的交叉点vector（54个三维坐标向量）构成gObjectPoints的一个元素
	//iSquareSize，标定板每个block的尺寸
	Size iSquareSize = Size(50, 50); //每个标定板方块的物理尺寸，我们的情况为50mm*50mm
	vector<vector<Point3f>>  gObjectPoints;
	gObjectPoints.clear();
	
	//gPointsCount是个向量，其维度为nImageCount，每个元素表示这张图像的交叉点的个数
	vector<int>  gPointsCount;
	//Generate 3d points.
	for (int t = 0; t < nImageCount; t++)
	{
		//gTempPointSet存储的是第t张标定板图像上所有交叉点的世界三维坐标
		vector<Point3f> gTempPointSet;
		for (int i = 0; i < iPatternSize.height; i++)
		{
			for (int j = 0; j < iPatternSize.width; j++)
			{
				Point3f iTempPoint;
				iTempPoint.x = i * iSquareSize.width;
				iTempPoint.y = j * iSquareSize.height;
				iTempPoint.z = 0;
				gTempPointSet.push_back(iTempPoint);
			}
		}
		gObjectPoints.push_back(gTempPointSet);
	}

	//记录每一张标定板图像上的交叉点个数，对我们的例子来说都是54
	for (int i = 0; i < nImageCount; i++)
	{
		gPointsCount.push_back(iPatternSize.width*iPatternSize.height);
	}

	Size iImageSize = gImages[0].size(); //图像分辨率
	int flags = 0;
	flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
	flags |= cv::fisheye::CALIB_CHECK_COND;
	flags |= cv::fisheye::CALIB_FIX_SKEW;
	cout << "开始标定\n";
	fisheye::calibrate(gObjectPoints, gAllCorners, iImageSize, mIntrinsicMatrix, mDistortion, 
		gRotationVectors, gTranslationVectors, flags, 
		cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 500, 1e-10));
	cout << "标定成功！\n";

	//根据相机参数，计算交叉点的重投影误差
	cout << "计算交叉点的重投影误差­" << endl;
	double nTotalError = 0.0; 
	vector<Point2f>  gImagePoints; //存储标定板交叉点的投影像素坐标            

	vector<double> gErrorVec; //其每个元素为每张标定板图像上所有交叉点重投影误差的平均值
	gErrorVec.reserve(nImageCount);
	for (int i = 0; i < nImageCount; i++)
	{
		//gTempPointSet，世界坐标系下当前标定板交叉点的三维坐标
		vector<Point3f> gTempPointSet = gObjectPoints[i];
		//根据相机内参数以及当前标定板外参数，计算该标定板上的交叉点在成像平面上的投影gImagePoints
		fisheye::projectPoints(gTempPointSet, gImagePoints, 
			gRotationVectors[i], gTranslationVectors[i], mIntrinsicMatrix, mDistortion);

		vector<Point2f> gTempImagePoint = gAllCorners[i]; //当前标定板图像中图像空间中交叉点坐标
		//计算观察到的图像空间中交叉点的像素坐标gTempImagePoint与
		//根据相机参数计算出来的交叉点重投影像素坐标gImagePoints之间的重投影误差
		Mat mTempImagePointMat = Mat(1, gTempImagePoint.size(), CV_32FC2);
		Mat mImagePoints2Mat = Mat(1, gImagePoints.size(), CV_32FC2);
		for (size_t i = 0; i != gTempImagePoint.size(); i++)
		{
			mImagePoints2Mat.at<Vec2f>(0, i) = Vec2f(gImagePoints[i].x, gImagePoints[i].y);
			mTempImagePointMat.at<Vec2f>(0, i) = Vec2f(gTempImagePoint[i].x, gTempImagePoint[i].y);
		}
		//计算两个向量之间误差的二范数
		double nError = norm(mImagePoints2Mat, mTempImagePointMat, NORM_L2); 
		nTotalError += nError /= gPointsCount[i];
		gErrorVec.push_back(nError);
		cout << strValidImgFileNames[i] << "的重投影误差为" << nError << endl;
	}
	cout << "平均重投影误差为 " << nTotalError / nImageCount << endl;
		
	cout << "测试一个图像的去畸变效果..." << endl;
	Mat iTestImage = gImages[0];
	Mat undistortedTestImg = iTestImage.clone();
	cv::fisheye::undistortImage(iTestImage, undistortedTestImg, mIntrinsicMatrix, mDistortion, mIntrinsicMatrix);
	cv::imshow("original", iTestImage);
	cv::waitKey(0);
	cv::imshow("undistortImage", undistortedTestImg);
	cv::waitKey(0);

	cout << "保存相机参数到本地文件 " << endl;
	camParamsFile << "intrinsic_matrix" << mIntrinsicMatrix;
	camParamsFile << "distortion_coefficients" << mDistortion;

	camParamsFile.release();
	cv::destroyAllWindows();
	return 0;
}

//main函数3
//该main函数应用之前已经获取到的相机内参，完成对输入视频的实时去畸变
//int main(int argc, char* argv[]) 
//{
//	//打开相机并设置好分辨率，注意：分辨率必须与标定时所用分辨率一致
//	VideoCapture inputVideo(0); 
//	inputVideo.set(cv::CAP_PROP_FRAME_HEIGHT, 768);
//	inputVideo.set(cv::CAP_PROP_FRAME_WIDTH, 1024);
//	if (!inputVideo.isOpened()) 
//	{
//		cout << "打开相机失败" << endl;
//		return -1;
//	}
//	else 
//	{
//		cout << "相机已经打开" << endl;
//	}
//
//	//该XML文件是之前相机标定操作之后存储相机内参的文件
//	FileStorage camPramsFile(dataDir + "\\camParams.xml", FileStorage::READ);
//	cv::Mat mIntrinsicMatrix, mDistortion;
//	camPramsFile["intrinsic_matrix"] >> mIntrinsicMatrix;
//	camPramsFile["distortion_coefficients"] >> mDistortion;
//	//输出相机内参数，内参矩阵与畸变系数
//	cout << "mIntrinsic is: " << endl << mIntrinsicMatrix << endl;
//	cout << "mDistortion is: " << endl << mDistortion << endl;
//
//	Mat oriframe; //原始输入视频帧
//	Mat shrinkedOriFrame, shrinkedUndistortedFrame; //为了便于屏幕显示，把帧尺寸缩小一些
//
//	while (1) 
//	{
//		inputVideo >> oriframe;
//		if (oriframe.empty())
//			continue;
//		Mat undistortFrame = oriframe.clone();
//		//对当前帧oriframe执行去畸变，结果存储在undistortFrame中
//		cv::fisheye::undistortImage(oriframe, undistortFrame, 
//			mIntrinsicMatrix, mDistortion, mIntrinsicMatrix);
//
//		//为方便显示，缩小原始输入视频帧为0.6倍
//		cv::resize(oriframe, shrinkedOriFrame,cv::Size(0,0), 0.6, 0.6); 
//		cv::resize(undistortFrame, shrinkedUndistortedFrame, cv::Size(0, 0), 0.6, 0.6);
//		imshow("原始视频", shrinkedOriFrame);
//		imshow("去畸变视频", shrinkedUndistortedFrame);
//		waitKey(20);
//	}
//	return 0;
//}
