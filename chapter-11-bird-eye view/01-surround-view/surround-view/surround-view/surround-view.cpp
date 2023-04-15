//�ó�����ʾ������������ڸ�����ʻ�������ͼ
//�����һ���ṩ�������ļ���������data�ļ����У�������߸���
//f.avi, l.avi��b.avi��r.avi���ĸ�ʱ��ͬ����������Ƶ�ļ��������ɰ�װ��ʵ�鳵���ϵ�ǰ���󡢺����ĸ������������õ������ǵ�Ŀ����Ҫ��
//���ĸ�������Ƶ�кϳɳ������ͼ
//homofor4cams.txt������ı��ļ��д洢�˴������ͼƽ�浽�ĸ��������ƽ�棨ȥ����֮�󣩵ĵ�Ӧ����������4��������������־���һ����4��
//���ĸ���Ӧ����Ľ�����Ҫ�����궨�������˹��궨�����ݵ�Թ�ϵͨ����С���˷��������
//intrinsics.xml������ļ��洢��4��������ڲ���������3*3�ڲξ����4����ͷ�������
//���ڴ������ͼƽ�浽�������ƽ��ĵ�Ӧ�����Լ�����ڲ������������ɳ��������ͼ��һ�㵽����ͼ����һ��Ĳ���ӳ�������
//�˲��ұ�����ʵʱ���ɳ������ͼ
//���֣�ͬ�ô�ѧ��2023��5��

#include "include/birdview.h"
using namespace std;

//�ú������ݴ������ͼ��ԭʼ����ͼ����ĸ�ӳ����Լ�ԭʼ�����������Ƶ�����������ͼ
void GenerateBirdView(string aFrontPathVideo , string aLeftPathVideo , string aBackPathVideo , string aRightPathVideo ,
					  string aFrontPath , string aLeftPath , string aBackPath , string aRightPath)
{
	//��ǰ������ĸ�Ԥ������õ���Ƶ�ļ�
	cv::VideoCapture iCaptureFront;
	iCaptureFront.open(aFrontPathVideo);
	cv::VideoCapture iCaptureLeft;
	iCaptureLeft.open(aLeftPathVideo);
	cv::VideoCapture iCaptureBack;
	iCaptureBack.open(aBackPathVideo);
	cv::VideoCapture iCaptureRight;
	iCaptureRight.open(aRightPathVideo);

	//�����ͼ�ֱ���
	cv::Size sWH(BIRDS_EYE_WIDTH , BIRDS_EYE_HEIGHT);
	//�����ĸ������ӳ���
	vector<vector<cv::Point2f>> gFrontMappingTable , gLeftMappingTable , gBackMappingTable ,gRightMappingTable;
	gFrontMappingTable = LoadMappingTable(aFrontPath , sWH);
	gLeftMappingTable = LoadMappingTable(aLeftPath , sWH);
	gBackMappingTable =  LoadMappingTable(aBackPath , sWH);
	gRightMappingTable = LoadMappingTable(aRightPath , sWH);
	cout << "���4��ӳ����ȡ" << endl;
	   
	int W = BIRDS_EYE_WIDTH , H = BIRDS_EYE_HEIGHT;
	
	//����ĸ��������ʼ���ĸ������ͼ���ֱ���������ƴ�ϺõĻ���ͼһ��
	//Ҫ�������ĸ������ͼ��Ȼ����ȷ����ƴ���ߵĻ����ϣ����ĸ������ͼƴ�ϳ����������ͼ
	cv::Mat iFrontBirdsEyeFrame , iLeftBirdsEyeFrame , iBackBirdsEyeFrame , iRightBirdsEyeFrame;
	cv::Mat iBirdsEyeImage= cv::Mat::zeros(W,H,CV_8UC3);
	iFrontBirdsEyeFrame = cv::Mat::zeros(W,H,CV_8UC3);
	iLeftBirdsEyeFrame = cv::Mat::zeros(W,H,CV_8UC3);
	iBackBirdsEyeFrame = cv::Mat::zeros(W,H,CV_8UC3);
	iRightBirdsEyeFrame = cv::Mat::zeros(W,H,CV_8UC3);
	   
	bool bStop = false;

    //ȷ�������·���������ɵĻ���ͼ�����ڱ���
	string outputVideoPath = dataDir + "surround-view-video.avi";
    cv::VideoWriter outputVideo;
    outputVideo.open(outputVideoPath, cv::VideoWriter::fourcc('M', 'P', '4', '2'), 25.0, sWH);

	cv::Mat iBirdsEyeImageCopy = iBirdsEyeImage.clone();
	while (!bStop)
	{
		//����·��Ƶ�У����뵱ǰ֡��½������gImages��
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

		//����ǰ�����ͼ
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

		//�����������ͼ
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

		//���ɺ������ͼ
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

		//�����������ͼ
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

		//���ϳ������ͼ������ǰ�ӵĲ���
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
		//���ϳ������ͼ���������ӵĲ���
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
		//���ϳ������ͼ�����Ժ��ӵĲ���
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
		//���ϳ������ͼ���������ӵĲ���
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
	cout << "��������ͼ���棡" << endl;
}

int main()
{
	//�ú������ڴ������ͼƽ�浽�������ƽ��ĵ�Ӧ�����Լ�����ڲ��������ɳ��������ͼ��һ�㵽����ͼ����һ��Ĳ���ӳ���
	//intrinsics.xml,�洢�ĸ���������ڲε��ļ�
	//homofor4cams.txt���洢�����ͼ���ĸ��������ƽ�棨ȥ����֮�󣩵ĵ�Ӧ����
	//���ұ�һ����4������Ӧ��4��������ֱ�����Ϊfront_table.txt��lef_table.txt,back_table.txt��right_table.txt
	GenerateMappingTable(dataDir + "intrinsics.xml", dataDir+ "homofor4cams.txt");
	cout << "ӳ�������������ɣ�" << endl;
	
	//�����ĸ���ͼ��ӳ����ұ��ԭʼ��·������Ƶ�����������ͼ
	GenerateBirdView(dataDir+ "f.avi", dataDir+"l.avi",
		             dataDir+"b.avi", dataDir + "r.avi",
		             dataDir+"front_table.txt", 
					 dataDir+"left_table.txt", 
				     dataDir+"back_table.txt", 
					 dataDir+"right_table.txt");
	
	return 0;
}
