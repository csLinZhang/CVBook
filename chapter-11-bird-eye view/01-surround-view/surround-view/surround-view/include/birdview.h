#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>
#include "config.h"

using namespace cv;
using namespace std;

//����������ͼ���������ƽ��ĵ�Ӧ����
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

//���ĳ����������ɴ������ͼ������ԭʼ����ͼ���ӳ���
//mHomography�������ͼ���������ƽ��ĵ�Ӧ����
//mK,mD���ֱ�Ϊ��������ڲξ���ͻ���ϵ��
//iImageSize, �����ͼ�ķֱ���
vector<vector<cv::Point2f>> GenerateSingleMappingTable(cv::Mat mHomography , cv::Size iImageSize , cv::Mat mK , cv::Mat mD )
{
	vector<vector<cv::Point2f>> gMappingTable;
	gMappingTable.reserve(iImageSize.height); //����ɨ��
	for (int y=0; y < iImageSize.height; y++)
	{	
		//������һ�е�ӳ����Ϣ
		vector<cv::Point2f> gSubMappingTable;
		gSubMappingTable.reserve(iImageSize.width);
		for (int x=0; x < iImageSize.width; x++)
		{
			cv::Mat mPoint = (cv::Mat_<double>(3 , 1) << x , y , 1); //�����ͼ��һ����������
			mPoint = mHomography * mPoint; //���ϵ�Ӧ����֮��ӳ�䵽�������ƽ���ϣ�ȥ���䣩һ��

			mPoint = mK.inv() * mPoint; //ӳ�䵽��һ������ƽ������ϵ֮��
			//iPoint����һ������ƽ������ϵ�ϵ���ͨ����
			cv::Point2f iPoint(mPoint.at<double>(0 , 0)/mPoint.at<double>(2 , 0) , mPoint.at<double>(1 , 0)/mPoint.at<double>(2 , 0));
			gSubMappingTable.push_back(iPoint);			
		}
		//�ڹ�һ������ƽ���϶Ե���л��������ע�⣺�õ��Ľ���������ճ���ƽ���ϵ�
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

//����������ͼ��ԭʼ����ͼ���ӳ����ұ�
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

//�ú������ڴ������ͼƽ�浽�������ƽ��ĵ�Ӧ�����Լ�����ڲ��������ɳ��������ͼ��һ�㵽����ͼ����һ��Ĳ���ӳ���
//aIntrinsicPath,�洢�ĸ���������ڲε��ļ�
//aHomographyPath���洢�����ͼ���ĸ��������ƽ�棨ȥ����֮�󣩵ĵ�Ӧ����
void GenerateMappingTable(string aIntrinsicPath , string aHomographyPath)
{
	//�����ĸ�������ڲ�����
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
	cout << "����ڲ����ݵ���ɹ���" << endl;

	cv::Size iBirdsEyeSize(BIRDS_EYE_HEIGHT, BIRDS_EYE_WIDTH);

	//����������ͼ���������ƽ��ĵ�Ӧ���������
	cv::Mat mHomographyFront , mHomographyLeft , mHomographyBack , mHomographyRight;
	vector<cv::Mat> gHomographyMatrices;
	gHomographyMatrices = ReadHomography(aHomographyPath);
	
	mHomographyFront = gHomographyMatrices[0];
	mHomographyLeft = gHomographyMatrices[1];
	mHomographyBack = gHomographyMatrices[2];
	mHomographyRight = gHomographyMatrices[3];
	cout << "��Ӧ�������ݵ���ɹ�!" << endl;

	vector<vector<cv::Point2f>> gMappingTableFront = GenerateSingleMappingTable(	mHomographyFront,
																				 	iBirdsEyeSize,
																				 	mFrontK,
																				 	mFrontD);
	cout << "ǰ�����ӳ������ɳɹ���" << endl;
	vector<vector<cv::Point2f>> gMappingTableLeft = GenerateSingleMappingTable(	mHomographyLeft,
																				 	iBirdsEyeSize,
																				 	mLeftK,
																				 	mLeftD);

	cout << "�������ӳ������ɳɹ���" << endl;
	vector<vector<cv::Point2f>> gMappingTableBack = GenerateSingleMappingTable(	mHomographyBack,
																				 	iBirdsEyeSize,
																				 	mBackK,
																				 	mBackD);

	cout << "�������ӳ������ɳɹ���" << endl;
	vector<vector<cv::Point2f>> gMappingTableRight = GenerateSingleMappingTable(	mHomographyRight,
																				 	iBirdsEyeSize,
																				 	mRightK,
																				 	mRightD);
	cout << "�������ӳ������ɳɹ���" << endl;

	
	string dataDir = "D:\\Self Made Books\\CV-Intro\\code\\chapter-11-bird-eye view\\01-surround-view\\surround-view\\data\\";

	SaveMappingTable(dataDir+ "front_table.txt", gMappingTableFront);
	cout << "ǰ�����ӳ�����ɹ���" << endl;
	SaveMappingTable(dataDir + "left_table.txt", gMappingTableLeft);
	cout << "�������ӳ�����ɹ���" << endl;
	SaveMappingTable(dataDir + "back_table.txt", gMappingTableBack);
	cout << "�������ӳ�����ɹ���" << endl;
	SaveMappingTable(dataDir+"right_table.txt", gMappingTableRight);
	cout << "�������ӳ�����ɹ���" << endl;
}


