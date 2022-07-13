
#include "./include/sift.h"
#include "./include/imgfeatures.h"
#include "./include/kdtree.h"
#include "./include/utils.h"
#include "./include/xform.h"

#include <opencv/cv.h>
#include <opencv/cxcore.h>
#include <opencv/highgui.h>

#include <stdio.h>

//本程序基于opensift，实现来自rob hess，http://robwhess.github.io/opensift/
//进行了部分修改，去掉了非主干部分的代码，更容易让读者学习SIFT实现的本质
//中文注释由同济大学张林撰写，2022年8月

/* the maximum number of keypoint NN candidates to check during BBF search */
#define KDTREE_BBF_MAX_NN_CHKS 200

/* threshold on squared ratio of distances between NN and 2nd NN */
#define NN_SQ_DIST_RATIO_THR 0.49


int main(int argc, char** argv)
{
	IplImage* img1, *img2, *stacked;
	struct feature* feat1, *feat2, *feat;
	struct feature** nbrs;
	struct kd_node* kd_root;
	CvPoint pt1, pt2;
	double d0, d1;
	int n1, n2, k, i, m = 0;

	//图像路径需要根据自己电脑路径进行修改！！！！
	img1 = cvLoadImage("C:\\lin\\CV-Intro\\code\\chapter-04-feature detection and matching\\03-openSIFTVS\\sse1.bmp", 1);
	img2 = cvLoadImage("C:\\lin\\CV-Intro\\code\\chapter-04-feature detection and matching\\03-openSIFTVS\\sse2.bmp", 1);

	stacked = stack_imgs(img1, img2);

	//利用SIFT算法，在图像img1上找出尺度不变特征点并计算它们的特征描述子
	n1 = sift_features(img1, &feat1, SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR,SIFT_CURV_THR, SIFT_DESCR_WIDTH,SIFT_DESCR_HIST_BINS);

	//export_features("C:\\lin\\CV-Intro\\code\\chapter-04-feature detection and matching\\03-openSIFTVS\\sse1.txt", feat1, n1);
	n2 = sift_features(img2, &feat2, SIFT_INTVLS, SIFT_SIGMA, SIFT_CONTR_THR,SIFT_CURV_THR, SIFT_DESCR_WIDTH,SIFT_DESCR_HIST_BINS);

	fprintf(stderr, "Building kd tree...\n");
	kd_root = kdtree_build(feat2, n2);
	for (i = 0; i < n1; i++)
	{
		feat = feat1 + i;
		k = kdtree_bbf_knn(kd_root, feat, 2, &nbrs, KDTREE_BBF_MAX_NN_CHKS);
		if (k == 2)
		{
			d0 = descr_dist_sq(feat, nbrs[0]);
			d1 = descr_dist_sq(feat, nbrs[1]);
			if (d0 < d1 * NN_SQ_DIST_RATIO_THR) //教材中的无歧义匹配原则
			{
				pt1 = cvPoint(cvRound(feat->x), cvRound(feat->y));
				pt2 = cvPoint(cvRound(nbrs[0]->x), cvRound(nbrs[0]->y));
				pt2.y += img1->height;
				cvLine(stacked, pt1, pt2, CV_RGB(255, 0, 255), 1, 8, 0);
				m++;
				feat1[i].fwd_match = nbrs[0];
			}
		}
		free(nbrs);
	}

	fprintf(stderr, "Found %d total matches\n", m);
	display_big_img(stacked, "Matches");
	cvWaitKey(0);

	/*
	   UNCOMMENT BELOW TO SEE HOW RANSAC FUNCTION WORKS

	   Note that this line above:

	   feat1[i].fwd_match = nbrs[0];

	   is important for the RANSAC function to work.
	*/
	/*
	{
	  CvMat* H;
	  IplImage* xformed;
	  H = ransac_xform( feat1, n1, FEATURE_FWD_MATCH, lsq_homog, 4, 0.01,
				homog_xfer_err, 3.0, NULL, NULL );
	  if( H )
		{
	  xformed = cvCreateImage( cvGetSize( img2 ), IPL_DEPTH_8U, 3 );
	  cvWarpPerspective( img1, xformed, H,
				 CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,
				 cvScalarAll( 0 ) );
	  cvNamedWindow( "Xformed", 1 );
	  cvShowImage( "Xformed", xformed );
	  cvWaitKey( 0 );
	  cvReleaseImage( &xformed );
	  cvReleaseMat( &H );
		}
	}
	*/

	cvReleaseImage(&stacked);
	cvReleaseImage(&img1);
	cvReleaseImage(&img2);
	kdtree_release(kd_root);
	free(feat1);
	free(feat2);
	return 0;
}


//int main(int argc, char** argv)
//{
//	IplImage* img;
//	struct feature* feat;
//	char* name;
//	int n;
//
//	img = cvLoadImage("C:\\lin\\CV-Intro\\code\\chapter-04-feature detection and matching\\03-openSIFTVS\\sse1.bmp", 1);
//	if (!img)
//		fatal_error("unable to load image from");
//	n = import_features("C:\\lin\\CV-Intro\\code\\chapter-04-feature detection and matching\\03-openSIFTVS\\sse1.txt", FEATURE_LOWE, &feat);
//	if (n == -1)
//		fatal_error("unable to import features from");
//	name = "sse.txt";
//
//	draw_features(img, feat, n);
//	cvNamedWindow(name, 1);
//	cvShowImage(name, img);
//	cvWaitKey(0);
//	return 0;
//}
