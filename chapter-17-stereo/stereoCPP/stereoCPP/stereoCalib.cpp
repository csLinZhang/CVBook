#include <iostream>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <io.h>

using namespace cv;
using namespace std;

//数据存放目录
static string dataDirName = "D:\\Books\\CV-Intro\\code\\chapter-17-stereo\\stereoCPP\\data\\";

//该函数完成双目系统的标定
static void
StereoCalib(const vector<string>& imagelist, Size boardSize, float squareSize, bool displayCorners = false, bool useCalibrated = true, bool showRectified = true)
{
	if (imagelist.size() % 2 != 0) //imagelist中存放的文件路径必须是左右成对的
	{
		cout << "Error: the image list contains odd (non-even) number of elements\n";
		return;
	}

	//存放从图像上检测的交叉点坐标，来自左目图像的交叉点坐标放在一个vector中，来自右目图像的交叉点放在另一个vector中
	vector<vector<Point2f> > imagePoints[2]; 
	//标定板上交叉点的世界坐标，世界坐标系由标定板自身建立
	vector<vector<Point3f> > objectPoints; 
	Size imageSize;

	int i, j, k, nimages = (int)imagelist.size() / 2; //标定板双目图像总计有多少对

	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	vector<string> goodImageList;

	for (i = j = 0; i < nimages; i++) //遍历双目图像对
	{
		for (k = 0; k < 2; k++) //对于每一对双目图像，遍历左右目
		{
			const string& filename = imagelist[i * 2 + k];
			Mat img = imread(filename, 0);
			if (img.empty())
				break;
			if (imageSize == Size())
				imageSize = img.size();
			bool found = false;
			vector<Point2f>& corners = imagePoints[k][j];

			//在图像上进行交叉点检测，检测到的交叉点坐标存入corners中
			found = findChessboardCorners(img, boardSize, corners,CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
				
			if (displayCorners) //是否需要显示交叉点检测结果
			{
				cout << filename << endl;
				Mat cimg, cimg1;
				cvtColor(img, cimg, COLOR_GRAY2BGR);
				drawChessboardCorners(cimg, boardSize, corners, found);
	
				string filenameforcornerdetectionimage = filename.substr(0, filename.length() - 4) + "_c.jpg";
				cout << "角点图像" << filenameforcornerdetectionimage << endl;
				imwrite(filenameforcornerdetectionimage, cimg);
				imshow(filenameforcornerdetectionimage, cimg);

				cv::waitKey();
				cv::destroyWindow(filenameforcornerdetectionimage);
			}

			//对检测到的角点位置进行进一步亚像素精化
			cornerSubPix(img, corners, Size(11, 11), Size(-1, -1),TermCriteria(TermCriteria::COUNT + TermCriteria::EPS,30, 0.01));
		}
		if (k == 2) //k=2时说明对于当前双目对来说已经处理完了它的左右目，可以把它们的路径存入可用双目对路径列表
		{
			goodImageList.push_back(imagelist[i * 2]);
			goodImageList.push_back(imagelist[i * 2 + 1]);
			j++;
		}
	}
	cout << j << " pairs have been successfully detected.\n";
	nimages = j; //此时的nimages指的是可用的双目图像对
	imagePoints[0].resize(nimages);
	imagePoints[1].resize(nimages);
	objectPoints.resize(nimages);

	for (i = 0; i < nimages; i++) //对于每一个双目图像对，构造它的标定板交叉点世界坐标
	{
		for (j = 0; j < boardSize.height; j++)
			for (k = 0; k < boardSize.width; k++)
				objectPoints[i].push_back(Point3f(k*squareSize, j*squareSize, 0));
	}

	cout << "Running stereo calibration ...\n"; //执行双目标定
	//对左右目相机进行单目内参标定，得到的内参值要作为双目标定过程中各自内参的初始化值；R、T在后面不会用到
	Mat cameraMatrix[2], distCoeffs[2],R,T; 
	double reproj_error = cv::calibrateCamera(objectPoints, imagePoints[0], imageSize, cameraMatrix[0], distCoeffs[0], R, T);
	cout << "Calibration error of the left camera:" << reproj_error<<endl;
	reproj_error = cv::calibrateCamera(objectPoints, imagePoints[1], imageSize, cameraMatrix[1], distCoeffs[1], R, T);
	cout << "Calibration error of the right camera:" << reproj_error << endl;
	
	//保存相机内参，分别是左右目相机的内参矩阵和畸变系数向量
	FileStorage fs(dataDirName + "intrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] << "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
		fs.release();
	}
	else
		cout << "Error: can not save the intrinsic parameters\n";

	//标定双目系统外参
	Mat E, F;
	double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1], cameraMatrix[0], distCoeffs[0], 
		cameraMatrix[1], distCoeffs[1],imageSize, R, T, E, F, CALIB_FIX_INTRINSIC);
	cout << "Calibration error of the stereo:" << rms << endl;
	
	Mat R1, R2, P1, P2, Q;
	Rect validRoi[2];

	//双目立体校正
	stereoRectify(cameraMatrix[0], distCoeffs[0],cameraMatrix[1], distCoeffs[1],imageSize, 
		R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);
	//R和T，双目相机间的外参
	//R1、R2为教材中所说的（左右）校正旋转矩阵
	//P1，将左校正相机坐标系下的一点投影到左校正化相机像素平面上（坐标为齐次坐标）
	//P2，将左校正相机坐标系下的一点投影到右校正化相机像素平面上（坐标为齐次坐标）
	//P1和P2矩阵含有左右校正化相机的内参信息
	//Q为从校正化左目图像上一点的信息（像素坐标及视差）计算其所对应的三维空间点坐标的投影矩阵，对应教材式17-38
	fs.open(dataDirName + "extrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";
		
	if (!showRectified)
		return;
	
	//以下代码是为了检验标定效果
	//用标定得到的参数计算校正化左右目图像，理想情况下得到的两幅结果图像是行对齐的
	//rmap[0][0]和rmap[0][1]，存储从校正化左目图像到原始左目图像的像素位置映射表
	//rmap[1][0]和rmap[1][1]，存储从校正化右目图像到原始右目图像的像素位置映射表
	Mat rmap[2][2];
	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	Mat canvas;
	double sf;
	int w, h;

	sf = 600. / MAX(imageSize.width, imageSize.height);
	w = cvRound(imageSize.width*sf);
	h = cvRound(imageSize.height*sf);
	canvas.create(h, w * 2, CV_8UC3);

	for (i = 0; i < nimages; i++)
	{
		for (k = 0; k < 2; k++)
		{
			Mat img = imread(goodImageList[i * 2 + k], 0), rimg, cimg;
			//根据映射表rmap，从原始采集的图像img中采样出校正化图像rimg
			remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
			cvtColor(rimg, cimg, COLOR_GRAY2BGR);
			Mat canvasPart = canvas(Rect(w*k, 0, w, h));
			resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
			if (useCalibrated)
			{
				Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
					cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
				rectangle(canvasPart, vroi, Scalar(0, 0, 255), 3, 8);
			}
		}

		for (j = 0; j < canvas.rows; j += 16)
			line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);

		imshow("rectified", canvas);
		char c = (char)waitKey();
		if (c == 27 || c == 'q' || c == 'Q')
			break;
	}
}

//filename是文本文件路径，该文件中存放了左右相机图像的存储地址
//该函数把这些地址读取出来，放入l之中
static bool readStringList(const string& filename, vector<string>& l)
{
	l.resize(0);
	ifstream storedtxtfile(filename, ios::in);
	string currentfilepath;
	while (getline(storedtxtfile, currentfilepath))
	{
		cout << currentfilepath << endl;
		l.push_back(currentfilepath);
	}
	storedtxtfile.close();
	return true;
}

//假设已经采集了成对的标定板双目图像，它们的存放路径已经写在了文本文件
//\chapter-17-stereo\stereoCPP\data\imgpaths.txt中
//该main函数完成双目相机标定，内参文件和外参文件被分别输出在
//\chapter-17-stereo\stereoCPP\data\intrinsics.yml
//\chapter-17-stereo\stereoCPP\data\extrinsics.yml
int main(int argc, char* argv[])
{
	Size boardSize(11, 8); //标定板上的交叉点维度
	//imagelistfn，存放标定板双目图像文件地址的文本文件
	string imagelistfn = dataDirName + "imgpaths.txt";

	bool showRectified = true;
	float squareSize = 50.0; //标定板上每个方格的大小，单位为毫米
	//imagelist，标定板双目图像文件路径列表
	vector<string> imagelist;
	bool ok = readStringList(imagelistfn, imagelist);
	if (!ok || imagelist.empty())
	{
		cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
	}
	//执行双目相机标定任务
	StereoCalib(imagelist, boardSize, squareSize, false, true, showRectified);
	return 0;
}