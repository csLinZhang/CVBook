
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
using namespace std;
using namespace cv;

int main(int argc , char* argv[])
{
    cout << "Begin to calibrate intrinsic!" << endl;

    //Whether show images with corners.
    bool bShowMode = false;

    cv::Size iPatternSize(9 , 6);

    string aFrontDirectoryName = "C:\\lin\\CV-Intro\\code\\chapter-10-imaging model and intrinsics calibration\\cameracalib\\cameracalib\\imgs\\";
    // string aLeftDirectoryName = argv[2];
    // string aBackDirectoryName = argv[3];
    // string aRightDirectoryName = argv[4];
    //Mapping table
    FileStorage fMappingTable("map4undistortLin.xml" , FileStorage::WRITE);
    //All directory names.
    vector<string> gDirectoryNames = {
                                        aFrontDirectoryName
                                        };

    //For all directories.
    for (int k=0;k<gDirectoryNames.size();k++){
        //The intrinsic matrix K
        cv::Mat mIntrinsicMatrix;
        //Distort coefficient.
        cv::Mat mDistortion;    
        //In fact these useless.
        std::vector<cv::Vec3d> gRotationVectors;
        std::vector<cv::Vec3d> gTranslationVectors;
        //Load images in this directory.
        string aDirectoryName = gDirectoryNames[k];
        vector<cv::String> gFileNames;
        //Load all files
        cv::glob(aDirectoryName, gFileNames);   
        cout << "Load images" << endl;
        for (auto aFileName : gFileNames){
            cout << aFileName << endl;
        }
        cout << gFileNames.size() << " images have been loaded" << endl;
        //Get the number of images.
        int nImageCount = gFileNames.size();
        //Pattern in 1 image.
        vector<Point2f> gCorners;
        //Pattern in all images.
        vector<vector<Point2f>>  gAllCorners;
        //All images in this directory.
        vector<Mat>  gImages;

        int nCornersCount = 0;
        //For all frames.
        for( int i = 0;  i < nImageCount ; i++)
        {
            cout<<"Frame #"<<i+1<<"..."<<endl;
            //Get the filiename of the image.
            string aImageFileName = gFileNames[i];
            //Load the image.
            cv::Mat iImage = imread(aImageFileName); 
            cout << "Filename " << aImageFileName << endl;   
            //Convert BGR to gray image.
            Mat iImageGray;        
            cvtColor(iImage, iImageGray , cv::COLOR_BGR2GRAY);
            bool bPatternFound = findChessboardCorners(iImage, iPatternSize, gCorners,CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE+ 
                CALIB_CB_FAST_CHECK );
            if (!bPatternFound)   
            {   
                cout<<"Can not find chessboard corners!\n";  
                continue;
            } 
            else
            {   
                //Sub pixel to get accurate position of corners.
                cornerSubPix(iImageGray, gCorners, Size(11, 11), Size(-1, -1), TermCriteria(cv::TermCriteria::EPS+ cv::TermCriteria::MAX_ITER, 30, 0.1));
                //Copy the image.
                Mat iImageTemp = iImage.clone();
                for (int j = 0; j < gCorners.size(); j++)
                {
                    //Draw the corners
                    circle( iImageTemp, gCorners[j], 10, Scalar(0,0,255), 2, 8, 0);
                }
                if (bShowMode){
                    cv::imshow("corners", iImageTemp);
                    cv::waitKey(1000);
                }
                //Save the image.
                // imwrite(aImageFileName,iImageTemp);
                cout<<"Frame corner#"<<i+1<<"...end"<<endl;
                //Count the corners.
                nCornersCount = nCornersCount + gCorners.size();
                gAllCorners.push_back(gCorners);
            }   
            gImages.push_back(iImage);
        }   
        bool bDeleteFlag = true;
        //Check if new images need to be deleted.
        while (bDeleteFlag){
            //Get the number of images.
            nImageCount = gImages.size();

            cout << "Now Image size " << nImageCount << endl; 
            //Generate 3d points.
            Size iSquareSize = Size(64.9,64.9);     
            vector<vector<Point3f>>  gObjectPoints;        


            gObjectPoints.clear();
            //Number of points. Use to calculate the mean reprojection error.
            vector<int>  gPointsCount;                                                         
            //Generate 3d points.
            for (int t = 0; t<nImageCount; t++)
            {
                vector<Point3f> gTempPointSet;
                for (int i = 0; i<iPatternSize.height; i++)
                {
                    for (int j = 0; j<iPatternSize.width; j++)
                    {
                        /* ¼ÙÉè¶¨±ê°å·ÅÔÚÊÀ½ç×ø±êÏµÖÐz=0µÄÆ½ÃæÉÏ */
                        Point3f iTempPoint;
                        iTempPoint.x = i*iSquareSize.width;
                        iTempPoint.y = j*iSquareSize.height;
                        iTempPoint.z = 0;
                        gTempPointSet.push_back(iTempPoint);
                    }
                }
                gObjectPoints.push_back(gTempPointSet);
            }
            for (int i = 0; i< nImageCount; i++)
            {
                gPointsCount.push_back(iPatternSize.width*iPatternSize.height);
            }
            
            Size iImageSize = gImages[0].size();
            
            int flags = 0;
            flags |= cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC;
            flags |= cv::fisheye::CALIB_CHECK_COND;
            flags |= cv::fisheye::CALIB_FIX_SKEW;
            cout<<"Begin to calibrate\n"; 
            fisheye::calibrate(gObjectPoints, gAllCorners, iImageSize, mIntrinsicMatrix, mDistortion, gRotationVectors, gTranslationVectors, flags, cv::TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 500, 1e-10));
            cout<<"Calibrate successfully!\n";   

            //Calculate reprojection error.

            cout<<"Calculate reprojection error­"<<endl;   
            double nTotalError = 0.0;                   /* ËùÓÐÍ¼ÏñµÄÆ½¾ùÎó²îµÄ×ÜºÍ */   
            vector<Point2f>  gImagePoints;             /****   ±£´æÖØÐÂ¼ÆËãµÃµ½µÄÍ¶Ó°µã    ****/   

            vector<double> gErrorVec;
            gErrorVec.reserve(nImageCount);
            for (int i=0;  i<nImageCount;  i++) 
            {
                vector<Point3f> gTempPointSet = gObjectPoints[i];
                //Reprojection.
                fisheye::projectPoints(gTempPointSet, gImagePoints, gRotationVectors[i], gTranslationVectors[i], mIntrinsicMatrix, mDistortion);
                
                vector<Point2f> gTempImagePoint = gAllCorners[i];
                Mat mTempImagePointMat = Mat(1,gTempImagePoint.size(),CV_32FC2);
                Mat mImagePoints2Mat = Mat(1,gImagePoints.size(), CV_32FC2);
                for (size_t i = 0 ; i != gTempImagePoint.size(); i++)
                {
                    mImagePoints2Mat.at<Vec2f>(0,i) = Vec2f(gImagePoints[i].x, gImagePoints[i].y);
                    mTempImagePointMat.at<Vec2f>(0,i) = Vec2f(gTempImagePoint[i].x, gTempImagePoint[i].y);
                }
                double nError = norm(mImagePoints2Mat, mTempImagePointMat, NORM_L2);
                nTotalError += nError/=  gPointsCount[i];   
                gErrorVec.push_back(nError);
                cout<<"No."<<i+1<<" reprojection error is "<<nError<<endl;   
            }
            cout<<"Mean error is "<<nTotalError/nImageCount<<endl;   
            double nMeanError = nTotalError/nImageCount;
            //Standard deviation
            double nStandardDeviation = 0.0;
            for (auto item : gErrorVec){
                nStandardDeviation += ((item - nMeanError) * (item - nMeanError));
            }   
            nStandardDeviation /= gErrorVec.size();
            nStandardDeviation = sqrt(nStandardDeviation);

            bDeleteFlag = false;
            int nLastSize = gErrorVec.size();
            for (int i=0;i<nImageCount;i++){
                if (gErrorVec[i] > nMeanError + 2 * nStandardDeviation){
                    cout << "Delete image " << "Error is " << gErrorVec[i] << " , Mean error is " << nMeanError << endl; 
                    gErrorVec.erase(gErrorVec.begin() + i);
                    gImages.erase(gImages.begin() + i);
                    gAllCorners.erase(gAllCorners.begin() + i);
                    i--;
                    nImageCount = gErrorVec.size();
                    bDeleteFlag = true;
                }
            }
            //Check how many images were deleted.
            cout << "Delete " << nLastSize-gErrorVec.size() << " images" << endl;
            cout << gImages.size() << " images left" << endl;
        }
    
        //Generate rectify map.

        Size iImageSize = gImages[0].size();
        iImageSize.height = iImageSize.height * 3.0;
        iImageSize.width = iImageSize.width * 3.0;
        Mat mapx = Mat(iImageSize,CV_32FC1);
        Mat mapy = Mat(iImageSize,CV_32FC1);
        Mat R = Mat::eye(3,3,CV_32F);


        cout<<"TestImage ..."<<endl;
        //Mat newCameraMatrix = Mat(3,3,CV_32FC1,Scalar::all(0));
        Mat iTestImage = gImages[0];
        fisheye::initUndistortRectifyMap(mIntrinsicMatrix,mDistortion,R,mIntrinsicMatrix,iImageSize,CV_32FC1,mapx,mapy);
        // MyinitUndistortRectifyMap(mIntrinsicMatrix, mDistortion, R, mIntrinsicMatrix, iImageSize, CV_32FC1, mapx, mapy, 4);
        Mat t = iTestImage.clone();
        // cv::remap(testImage,t,mapx, mapy, INTER_LINEAR);

        cv::fisheye::undistortImage(iTestImage, t, mIntrinsicMatrix, mDistortion, mIntrinsicMatrix);
        cv::imshow("original", iTestImage);
        cv::waitKey(0);
        cv::imshow("undistortImage",t);
        cv::waitKey(0);

        cout << "Writing mapping tables " << endl;
        switch (k){
            case 0: {
                fMappingTable << "f_intrinsic" << mIntrinsicMatrix;
                fMappingTable << "f_distortion" << mDistortion;
                
                break;
            }
            case 1: {
                fMappingTable << "l_intrinsic" << mIntrinsicMatrix;
                fMappingTable << "l_distortion" << mDistortion;
                
                break;
            }
            case 2: {
                fMappingTable << "b_intrinsic" << mIntrinsicMatrix;
                fMappingTable << "b_distortion" << mDistortion;
                
                break;
            }
            case 3: {
                fMappingTable << "r_intrinsic" << mIntrinsicMatrix;
                fMappingTable << "r_distortion" << mDistortion;
                
                break;
            }
        }
    }
        
	
	//Mat b_mapx, b_mapy, f_mapx, f_mapy, l_mapx, l_mapy, r_mapx, r_mapy;
	fMappingTable.release();
    cv::destroyAllWindows();

    return 0;
}
