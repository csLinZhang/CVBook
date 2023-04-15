//该程序演示了如何生成用于辅助驾驶的鸟瞰环视图
//随程序一起提供了数据文件，放在了data文件夹中，方便读者复现
//f.avi, l.avi，b.avi和r.avi是四个时间同步的鱼眼视频文件，它们由安装在实验车辆上的前、左、后、右四个鱼眼相机拍摄得到，我们的目标是要从
//这四个鱼眼视频中合成出鸟瞰环视图
//homofor4cams.txt，这个文本文件中存储了从鸟瞰视图平面到四个相机成像平面（去畸变之后）的单应矩阵，由于有4个相机，所以这种矩阵一共有4个
//这四个单应矩阵的建立需要借助标定场进行人工标定，根据点对关系通过最小二乘法计算出来
//intrinsics.xml，这个文件存储了4个相机的内参数，包括3*3内参矩阵和4个镜头畸变参数
//基于从鸟瞰视图平面到相机成像平面的单应矩阵，以及相机内参数，可以生成出从鸟瞰视图上一点到鱼眼图像上一点的查找映射表，基于
//此查找表便可以实时生成出鸟瞰环视图
//张林，同济大学，2023年5月

#include "include/birdview.h"
using namespace std;

//该函数根据从鸟瞰视图到原始鱼眼图像的四个映射表，以及原始拍摄的鱼眼视频，生成鸟瞰环视图
void GenerateBirdView(string aFrontPathVideo , string aLeftPathVideo , string aBackPathVideo , string aRightPathVideo ,
					  string aFrontPath , string aLeftPath , string aBackPath , string aRightPath)
{
	//打开前左后右四个预先拍摄好的视频文件
	cv::VideoCapture iCaptureFront;
	iCaptureFront.open(aFrontPathVideo);
	cv::VideoCapture iCaptureLeft;
	iCaptureLeft.open(aLeftPathVideo);
	cv::VideoCapture iCaptureBack;
	iCaptureBack.open(aBackPathVideo);
	cv::VideoCapture iCaptureRight;
	iCaptureRight.open(aRightPathVideo);

	//鸟瞰视图分辨率
	cv::Size sWH(BIRDS_EYE_WIDTH , BIRDS_EYE_HEIGHT);
	//读入四个相机的映射表
	vector<vector<cv::Point2f>> gFrontMappingTable , gLeftMappingTable , gBackMappingTable ,gRightMappingTable;
	gFrontMappingTable = LoadMappingTable(aFrontPath , sWH);
	gLeftMappingTable = LoadMappingTable(aLeftPath , sWH);
	gBackMappingTable =  LoadMappingTable(aBackPath , sWH);
	gRightMappingTable = LoadMappingTable(aRightPath , sWH);
	cout << "完成4个映射表读取" << endl;
	   
	int W = BIRDS_EYE_WIDTH , H = BIRDS_EYE_HEIGHT;
	
	//针对四个相机，初始化四个鸟瞰视图，分辨率与最终拼合好的环视图一致
	//要先生成四个鸟瞰视图，然后在确定好拼接线的基础上，从四个鸟瞰视图拼合成最终鸟瞰环视图
	cv::Mat iFrontBirdsEyeFrame , iLeftBirdsEyeFrame , iBackBirdsEyeFrame , iRightBirdsEyeFrame;
	cv::Mat iBirdsEyeImage= cv::Mat::zeros(W,H,CV_8UC3);
	iFrontBirdsEyeFrame = cv::Mat::zeros(W,H,CV_8UC3);
	iLeftBirdsEyeFrame = cv::Mat::zeros(W,H,CV_8UC3);
	iBackBirdsEyeFrame = cv::Mat::zeros(W,H,CV_8UC3);
	iRightBirdsEyeFrame = cv::Mat::zeros(W,H,CV_8UC3);
	   
	bool bStop = false;

    //确定好输出路径，将生成的环视图保存在本地
	string outputVideoPath = dataDir + "surround-view-video.avi";
    cv::VideoWriter outputVideo;
    outputVideo.open(outputVideoPath, cv::VideoWriter::fourcc('M', 'P', '4', '2'), 25.0, sWH);

	cv::Mat iBirdsEyeImageCopy = iBirdsEyeImage.clone();
	while (!bStop)
	{
		//从四路视频中，读入当前帧并陆续存入gImages中
		vector<cv::Mat> gImages;
		cv::Mat iFrontFrame , iLeftFrame , iBackFrame , iRightFrame;
		bool bRes1 = iCaptureFront.read(iFrontFrame);
		bool bRes2 = iCaptureLeft.read(iLeftFrame);
		bool bRes3 = iCaptureBack.read(iBackFrame);
		bool bRes4 = iCaptureRight.read(iRightFrame);
		if (!bRes1 || !bRes2 || !bRes3 || !bRes4)
		{
			break;
		}
		gImages.push_back(iFrontFrame);
		gImages.push_back(iLeftFrame);
		gImages.push_back(iBackFrame);
		gImages.push_back(iRightFrame);		

		//生成前鸟瞰视图
		int maxV = (BIRDS_EYE_HEIGHT-CENTER_HEIGHT)/2;
		int minV = 0;
		int maxU = (BIRDS_EYE_WIDTH/4*3);
		int minU = BIRDS_EYE_WIDTH/4;
		for (int v=0;v<= maxV;v++)
		{
				for (int u= minU;u<= maxU;u++)
				{
                    //Front area       
					cv::Point2f iMapping = gFrontMappingTable[v][u];
					iFrontBirdsEyeFrame.at<cv::Vec3b>(v , u) = gImages[0].at<cv::Vec3b>((int)iMapping.y , (int) iMapping.x);
	            }
	    }

		//生成左鸟瞰视图
	    maxV = BIRDS_EYE_HEIGHT;
	    maxU = BIRDS_EYE_WIDTH/2;
	    int subMaxV = (BIRDS_EYE_HEIGHT - CENTER_HEIGHT)/2;
	    int subMinV = (BIRDS_EYE_HEIGHT + CENTER_HEIGHT)/2;
	    int subMaxU = (BIRDS_EYE_WIDTH - CENTER_WIDTH)/2;
		for (int v=0;v< maxV;v++)
		{
				for (int u=0;u<=maxU;u++)
				{
	                    //Left area.
						cv::Point2f iMapping = gLeftMappingTable[v][u];
						iLeftBirdsEyeFrame.at<cv::Vec3b>(v , u) = gImages[1].at<cv::Vec3b>((int)iMapping.y , (int) iMapping.x);	
	            }
	    }

		//生成后鸟瞰视图
	    minV = (BIRDS_EYE_HEIGHT + CENTER_HEIGHT)/2;
	    maxV = BIRDS_EYE_HEIGHT;
	    minU = BIRDS_EYE_WIDTH/4;
	    maxU = (BIRDS_EYE_WIDTH/4*3);
		for (int v= minV;v< maxV;v++)
		{
				for (int u= minU;u<= maxU;u++)
				{
                    //Back area.    
					cv::Point2f iMapping = gBackMappingTable[v][u];
					iBackBirdsEyeFrame.at<cv::Vec3b>(v , u) = gImages[2].at<cv::Vec3b>((int)iMapping.y , (int) iMapping.x);	
	            }
	    }

		//生成右鸟瞰视图
	    minU = BIRDS_EYE_WIDTH/2;
	    subMinV = (BIRDS_EYE_WIDTH + CENTER_WIDTH)/2;
	    subMaxV = (BIRDS_EYE_HEIGHT - CENTER_HEIGHT)/2;
	    int subMinU = (BIRDS_EYE_WIDTH + CENTER_WIDTH)/2;
		for (int v=0;v<BIRDS_EYE_HEIGHT;v++)
		{
				for (int u= minU;u<BIRDS_EYE_WIDTH;u++)
				{	
        			//Right area.
					cv::Point2f iMapping = gRightMappingTable[v][u];
					iRightBirdsEyeFrame.at<cv::Vec3b>(v , u) = gImages[3].at<cv::Vec3b>((int)iMapping.y , (int) iMapping.x);
	            }
	    }

		//填充合成鸟瞰环视图中来自前视的部分
		maxV = (BIRDS_EYE_HEIGHT-CENTER_HEIGHT)/2;
		minV = 0;
		maxU = (BIRDS_EYE_WIDTH/4*3);
		minU = BIRDS_EYE_WIDTH/4;
		for (int v=0;v<= maxV;v++)
		{
				for (int u= minU;u<= maxU;u++)
				{
		            //Front area                                                
						iBirdsEyeImage.at<cv::Vec3b>(v , u) = iFrontBirdsEyeFrame.at<cv::Vec3b>(v , u);
				}
	    }
		//填充合成鸟瞰环视图中来自左视的部分
		maxV = BIRDS_EYE_HEIGHT;
	    maxU = BIRDS_EYE_WIDTH/2;

	    subMaxV = (BIRDS_EYE_HEIGHT - CENTER_HEIGHT)/2;
	    subMinV = (BIRDS_EYE_HEIGHT + CENTER_HEIGHT)/2;
	    subMaxU = (BIRDS_EYE_WIDTH - CENTER_WIDTH)/2;
		for (int v=0; v<maxV; v++)
		{
				for (int u=0; u<=subMaxU; u++)
				{
			        if (u<((BIRDS_EYE_WIDTH)/4+v/2) && u <= ((BIRDS_EYE_WIDTH+2*BIRDS_EYE_HEIGHT)/4-v/2))
					{
			                //Left area.            	
							iBirdsEyeImage.at<cv::Vec3b>(v , u) = iLeftBirdsEyeFrame.at<cv::Vec3b>(v , u);
			    	}
			}
	    }
		//填充合成鸟瞰环视图中来自后视的部分
 		minV = (BIRDS_EYE_HEIGHT + CENTER_HEIGHT)/2;
	    maxV = BIRDS_EYE_HEIGHT;
	    minU = BIRDS_EYE_WIDTH/4;
	    maxU = (BIRDS_EYE_WIDTH/4*3);
		for (int v= minV;v< maxV;v++)
		{
				for (int u= minU;u<= maxU;u++)
				{
			        if (u < ((BIRDS_EYE_WIDTH)/4+v/2) && u > ((BIRDS_EYE_WIDTH+2*BIRDS_EYE_HEIGHT)/4-v/2))
					{
			                //Back area.    
							iBirdsEyeImage.at<cv::Vec3b>(v , u) = iBackBirdsEyeFrame.at<cv::Vec3b>(v , u);
			    	}
			    }
	    }
		//填充合成鸟瞰环视图中来自右视的部分
        minU = BIRDS_EYE_WIDTH/2;
	    subMinV = (BIRDS_EYE_WIDTH + CENTER_WIDTH)/2;
	    subMaxV = (BIRDS_EYE_HEIGHT - CENTER_HEIGHT)/2;
	    subMinU = (BIRDS_EYE_WIDTH + CENTER_WIDTH)/2;
		for (int v=0;v<BIRDS_EYE_HEIGHT;v++)
		{
			for (int u= subMinU;u<BIRDS_EYE_WIDTH;u++)
			{	   		
		        if (u>=((BIRDS_EYE_WIDTH)/4+v/2) && u > ((BIRDS_EYE_WIDTH+2*BIRDS_EYE_HEIGHT)/4-v/2)) 
				{
		            	//Right area.
						iBirdsEyeImage.at<cv::Vec3b>(v , u) = iRightBirdsEyeFrame.at<cv::Vec3b>(v , u);	
		        }
	        }
	    }

		cv::imshow("show", iBirdsEyeImage);
		outputVideo << iBirdsEyeImage;

		//Clear the images.
		gImages.clear();
		if (cv::waitKey(10) >= 0)
		{
			bStop = true;		   	
		}
	}
	outputVideo.release();
	cout << "完成鸟瞰环视图保存！" << endl;
}

int main()
{
	//该函数基于从鸟瞰视图平面到相机成像平面的单应矩阵以及相机内参数，生成出从鸟瞰视图上一点到鱼眼图像上一点的查找映射表
	//intrinsics.xml,存储四个鱼眼相机内参的文件
	//homofor4cams.txt，存储鸟瞰视图到四个相机成像平面（去畸变之后）的单应矩阵
	//查找表一共有4个，对应于4个相机，分别命名为front_table.txt，lef_table.txt,back_table.txt和right_table.txt
	GenerateMappingTable(dataDir + "intrinsics.xml", dataDir+ "homofor4cams.txt");
	cout << "映射表生成任务完成！" << endl;
	
	//根据四个视图的映射查找表和原始四路鱼眼视频，生成鸟瞰环视图
	GenerateBirdView(dataDir+ "f.avi", dataDir+"l.avi",
		             dataDir+"b.avi", dataDir + "r.avi",
		             dataDir+"front_table.txt", 
					 dataDir+"left_table.txt", 
				     dataDir+"back_table.txt", 
					 dataDir+"right_table.txt");
	
	return 0;
}
