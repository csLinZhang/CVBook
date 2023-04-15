#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include "config.h"

using namespace cv;
using namespace std;

//导入从鸟瞰视图到相机成像平面的单应矩阵
vector<cv::Mat> ReadHomography(string aFileName)
{
	cv::Mat mHomography;
	vector<cv::Mat> gHomographyMatrices;
	gHomographyMatrices.reserve(4);
	ifstream fIn(aFileName);
	for (int i=0;i<4;i++)
	{
		string s;
		getline(fIn, s);
		cv::Mat_<double> iMat(3 , 3);
		for (int j=0;j<9;j++)
		{
			double value;
			fIn >> value;
			iMat.at<double>(j/3 , j%3) = value;
		}
		mHomography = iMat;
		gHomographyMatrices.push_back(mHomography);
		getline(fIn, s);			
	}
	fIn.close();
	return gHomographyMatrices;
}

//针对某个相机，生成从鸟瞰视图到最终原始鱼眼图像的映射表
//mHomography，鸟瞰视图到相机成像平面的单应矩阵
//mK,mD，分别为该相机的内参矩阵和畸变系数
//iImageSize, 鸟瞰环视图的分辨率
vector<vector<cv::Point2f>> GenerateSingleMappingTable(cv::Mat mHomography , cv::Size iImageSize , cv::Mat mK , cv::Mat mD )
{
	vector<vector<cv::Point2f>> gMappingTable;
	gMappingTable.reserve(iImageSize.height); //逐行扫描
	for (int y=0; y < iImageSize.height; y++)
	{	
		//生成这一行的映射信息
		vector<cv::Point2f> gSubMappingTable;
		gSubMappingTable.reserve(iImageSize.width);
		for (int x=0; x < iImageSize.width; x++)
		{
			cv::Mat mPoint = (cv::Mat_<double>(3 , 1) << x , y , 1); //鸟瞰视图这一点的齐次坐标
			mPoint = mHomography * mPoint; //乘上单应矩阵之后，映射到相机成像平面上（去畸变）一点

			mPoint = mK.inv() * mPoint; //映射到归一化成像平面坐标系之下
			//iPoint，归一化成像平面坐标系上的普通坐标
			cv::Point2f iPoint(mPoint.at<double>(0 , 0)/mPoint.at<double>(2 , 0) , mPoint.at<double>(1 , 0)/mPoint.at<double>(2 , 0));
			gSubMappingTable.push_back(iPoint);			
		}
		//在归一化成像平面上对点进行畸变操作。注意：得到的结果是在最终成像平面上的
		cv::fisheye::distortPoints(gSubMappingTable, gSubMappingTable, mK, mD);
		for (auto & item : gSubMappingTable)
		{
			if (item.x <= 0.0)
			{
				item.x = 0.0;
			}
			else if (item.x >= (float)(iCameraImageSize.width-1))
			{
				item.x = (float)(iCameraImageSize.width-1);
			}
			if (item.y <=0.0)
			{
				item.y = 0.0;
			}
			else if (item.y >= (float)(iCameraImageSize.height-1))
			{
				item.y = (float)(iCameraImageSize.height-1);
			}
		}
		gMappingTable.push_back(gSubMappingTable);
	}
	return gMappingTable;
}

//保存从鸟瞰视图到原始鱼眼图像的映射查找表
bool SaveMappingTable(string aFileName , vector<vector<cv::Point2f>> gMappingTable)
{
	ofstream fOut(aFileName , fstream::out);
	for (auto item : gMappingTable)
	{
		for (auto iPoint : item)
		{
			fOut << iPoint << endl;
		}
	}
	fOut.close();
	return true;
}

vector<vector<cv::Point2f>> LoadMappingTable(string aFileName , cv::Size iImageSize)
{
	ifstream fIn(aFileName , fstream::in);
	vector<vector<cv::Point2f>> gMappingTable;
	gMappingTable.reserve(iImageSize.height);
	for (int y=0;y<iImageSize.height;y++)
	{
		vector<cv::Point2f> gSubTable;
		gSubTable.reserve(iImageSize.width);
		for (int x=0;x<iImageSize.width;x++)
		{
			cv::Point2f iPoint;
			string s;
			getline(fIn, s);
			float xx , yy;
			sscanf_s(s.data(), "[%f, %f]", &xx , &yy);
			iPoint.x = (double)xx;
			iPoint.y = (double)yy;
			gSubTable.push_back(iPoint);
		}
		gMappingTable.push_back(gSubTable);
	}
	return gMappingTable;
}

//该函数基于从鸟瞰视图平面到相机成像平面的单应矩阵以及相机内参数，生成出从鸟瞰视图上一点到鱼眼图像上一点的查找映射表
//aIntrinsicPath,存储四个鱼眼相机内参的文件
//aHomographyPath，存储鸟瞰视图到四个相机成像平面（去畸变之后）的单应矩阵
void GenerateMappingTable(string aIntrinsicPath , string aHomographyPath)
{
	//导入四个相机的内参数据
	FileStorage fs(aIntrinsicPath, FileStorage::READ);
	cv::Mat mFrontK , mFrontD;
	cv::Mat mLeftK , mLeftD;
	cv::Mat mBackK , mBackD;
	cv::Mat mRightK , mRightD;

	fs["f_intrinsic"] >> mFrontK;
	fs["f_distortion"] >> mFrontD;

	fs["l_intrinsic"] >> mLeftK;
	fs["l_distortion"] >> mLeftD;

	fs["b_intrinsic"] >> mBackK;
	fs["b_distortion"] >> mBackD;

	fs["r_intrinsic"] >> mRightK;
	fs["r_distortion"] >> mRightD;
	fs.release();
	cout << "相机内参数据导入成功！" << endl;

	cv::Size iBirdsEyeSize(BIRDS_EYE_HEIGHT, BIRDS_EYE_WIDTH);

	//导入从鸟瞰视图到相机成像平面的单应矩阵的数据
	cv::Mat mHomographyFront , mHomographyLeft , mHomographyBack , mHomographyRight;
	vector<cv::Mat> gHomographyMatrices;
	gHomographyMatrices = ReadHomography(aHomographyPath);
	
	mHomographyFront = gHomographyMatrices[0];
	mHomographyLeft = gHomographyMatrices[1];
	mHomographyBack = gHomographyMatrices[2];
	mHomographyRight = gHomographyMatrices[3];
	cout << "单应矩阵数据导入成功!" << endl;

	vector<vector<cv::Point2f>> gMappingTableFront = GenerateSingleMappingTable(	mHomographyFront,
																				 	iBirdsEyeSize,
																				 	mFrontK,
																				 	mFrontD);
	cout << "前视相机映射表生成成功！" << endl;
	vector<vector<cv::Point2f>> gMappingTableLeft = GenerateSingleMappingTable(	mHomographyLeft,
																				 	iBirdsEyeSize,
																				 	mLeftK,
																				 	mLeftD);

	cout << "左视相机映射表生成成功！" << endl;
	vector<vector<cv::Point2f>> gMappingTableBack = GenerateSingleMappingTable(	mHomographyBack,
																				 	iBirdsEyeSize,
																				 	mBackK,
																				 	mBackD);

	cout << "后视相机映射表生成成功！" << endl;
	vector<vector<cv::Point2f>> gMappingTableRight = GenerateSingleMappingTable(	mHomographyRight,
																				 	iBirdsEyeSize,
																				 	mRightK,
																				 	mRightD);
	cout << "右视相机映射表生成成功！" << endl;

	
	string dataDir = "D:\\Self Made Books\\CV-Intro\\code\\chapter-11-bird-eye view\\01-surround-view\\surround-view\\data\\";

	SaveMappingTable(dataDir+ "front_table.txt", gMappingTableFront);
	cout << "前视相机映射表保存成功！" << endl;
	SaveMappingTable(dataDir + "left_table.txt", gMappingTableLeft);
	cout << "左视相机映射表保存成功！" << endl;
	SaveMappingTable(dataDir + "back_table.txt", gMappingTableBack);
	cout << "后视相机映射表保存成功！" << endl;
	SaveMappingTable(dataDir+"right_table.txt", gMappingTableRight);
	cout << "右视相机映射表保存成功！" << endl;
}


