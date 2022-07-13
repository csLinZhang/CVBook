/*
  Functions for detecting SIFT image features.
  
  For more information, refer to:
  
  Lowe, D.  Distinctive image features from scale-invariant keypoints.
  <EM>International Journal of Computer Vision, 60</EM>, 2 (2004),
  pp.91--110.
  
  Copyright (C) 2006-2012  Rob Hess <rob@iqengines.com>

  Note: The SIFT algorithm is patented in the United States and cannot be
  used in commercial products without a license from the University of
  British Columbia.  For more information, refer to the file LICENSE.ubc
  that accompanied this distribution.

  @version 1.1.2-20100521
*/

#include "../include/sift.h"
#include "../include/imgfeatures.h"
#include "../include/utils.h"

#include <opencv/cxcore.h>
#include <opencv/cv.h>

/************************* Local Function Prototypes *************************/

static IplImage* create_init_img( IplImage*, double );
static IplImage* convert_to_gray32( IplImage* );
static IplImage*** build_gauss_pyr( IplImage*, int, int, double );
static IplImage* downsample( IplImage* );
static IplImage*** build_dog_pyr( IplImage***, int, int );
static CvSeq* scale_space_extrema( IplImage***, int, int, double, int,
				   CvMemStorage*);
static int is_extremum( IplImage***, int, int, int, int );
static struct feature* interp_extremum( IplImage***, int, int, int, int, int,
					double);
static void interp_step( IplImage***, int, int, int, int, double*, double*,
			 double* );
static CvMat* deriv_3D( IplImage***, int, int, int, int );
static CvMat* hessian_3D( IplImage***, int, int, int, int );
static double interp_contr( IplImage***, int, int, int, int, double, double,
			    double );
static struct feature* new_feature( void );
static int is_too_edge_like( IplImage*, int, int, int );
static void calc_feature_scales( CvSeq*, double, int );
static void adjust_for_img_dbl( CvSeq* );
static void calc_feature_oris( CvSeq*, IplImage*** );
static double* ori_hist( IplImage*, int, int, int, int, double );
static int calc_grad_mag_ori( IplImage*, int, int, double*, double* );
static void smooth_ori_hist( double*, int );
static double dominant_ori( double*, int );
static void add_good_ori_features( CvSeq*, double*, int, double, struct feature* );
static struct feature* clone_feature( struct feature* );
static void compute_descriptors( CvSeq*, IplImage***, int, int );
static double*** descr_hist( IplImage*, int, int, double, double, int, int );
static void interp_hist_entry( double***, double, double, double, double, int,
			       int);
static void hist_to_descr( double***, int, int, struct feature* );
static void normalize_descr( struct feature* );
static int feature_cmp( void*, void*, void* );
static void release_descr_hist( double****, int );
static void release_pyr( IplImage****, int, int );


/*********************** Functions prototyped in sift.h **********************/
/**
   对输入图像img提取尺度不变特征点并计算SIFT特征
   该函数的返回值为img上尺度不变的点的个数

   img，输入图像
   feat，指针数组，存储SIFT特征描述符
   intvls，高斯尺度空间中，每组的层数，对应于教材中的s，每组最终层数为intvls+3
   sigma，每组第0层的高斯尺度
   @param cont_thr a threshold on the value of the scale space function
     \f$\left|D(\hat{x})\right|\f$, where \f$\hat{x}\f$ is a vector specifying
     feature location and scale, used to reject unstable features;  assumes
     pixel values in the range [0, 1]
   @param curv_thr threshold on a feature's ratio of principle curvatures
     used to reject features that are too edge-like
   descr_width，把用于计算SIFT描述子的局部区域划分成descr_width*descr_width的块，这个值为4
   descr_hist_bins，每个小块内梯度方向直方图小仓（bin）的数目，这个值为8

   返回这个图像img中成功检测到的特征点的个数
*/
extern int sift_features( IplImage* img, struct feature** feat, int intvls, double sigma, double contr_thr, int curv_thr,
	int descr_width, int descr_hist_bins)
{
	//高斯尺度空间中的初始图像，其分辨率为img的2倍，对应高斯尺度为1.6
	IplImage* init_img; 
	IplImage*** gauss_pyr, *** dog_pyr;
	CvMemStorage* storage;
	CvSeq* features;
	int octvs, i, n = 0;

	/* check arguments */
	if (!img)
		fatal_error("NULL pointer error, %s, line %d", __FILE__, __LINE__);
	if (!feat)
		fatal_error("NULL pointer error, %s, line %d", __FILE__, __LINE__);

	//初始化高斯尺度空间的初始层，sigma为初始层的尺度为1.6
	init_img = create_init_img(img, sigma);
	//确定高斯尺度空间的组数，因为每组是上一组分辨率的一半，因此组数是有限的且与图像的短边长度有关
	octvs = log(MIN(init_img->width, init_img->height)) / log(2) - 2;
	//构建高斯尺度空间金字塔，octvs组数，intvls为每组层数，sigma为初始层高斯尺度
	gauss_pyr = build_gauss_pyr(init_img, octvs, intvls, sigma);
	//构建DoG尺度空间
	dog_pyr = build_dog_pyr(gauss_pyr, octvs, intvls);

	storage = cvCreateMemStorage(0);
	
	//在DoG尺度空间金字塔中寻找尺度不变特征点
	features = scale_space_extrema(dog_pyr, octvs, intvls, contr_thr, curv_thr, storage);
	//计算特征点的尺度，由于我们之前已经知道了它所在的层以及层的偏移量，可以插值计算出精确的尺度值
	calc_feature_scales(features, sigma, intvls);
	
	//因为在构建尺度空间时，初始图像的分辨率提升至了输入图像的2倍，而我们最终的特征点位置是以输入图像为基准的，因此需要调整一下特征点的位置和尺度
	adjust_for_img_dbl(features);

	//到这一步为止，features中存储了特征点相对于输入图像分辨率的位置，还有它的特征尺度。
	//也存储了该特征点在DoG尺度空间中的位置，[octIndex, layIndex, r, c]
	//这一步要计算特征点的主方向，如果某个特征点有多于一个的主方向的话，需要复制几份，每个都当作独立的特征点进行后续处理
	//也就是说，在某个位置、某个特征尺度上，有可能存在几个具有不同主方向的特征点
	calc_feature_oris(features, gauss_pyr);

	//目前，features数组当中的每个元素存储了特征点的位置，特征尺度，特征尺度所在组和层，还有主方向信息
	//现在基于这些信息要为每个特征点构建描述子
	//descr_width值为4，descr_hist_bins值为8，这样SIFT描述子的维度就是4*4*8=128
	compute_descriptors(features, gauss_pyr, descr_width, descr_hist_bins);

	/* sort features by decreasing scale and move from CvSeq to array */
	cvSeqSort(features, (CvCmpFunc)feature_cmp, NULL);
	n = features->total;
	*feat = calloc(n, sizeof(struct feature));
	*feat = cvCvtSeqToArray(features, *feat, CV_WHOLE_SEQ);
	for (i = 0; i < n; i++)
	{
		free((*feat)[i].feature_data);
		(*feat)[i].feature_data = NULL;
	}

	cvReleaseMemStorage(&storage);
	cvReleaseImage(&init_img);
	release_pyr(&gauss_pyr, octvs, intvls + 3);
	release_pyr(&dog_pyr, octvs, intvls + 2);
	return n;
}

/*
  该函数用于准备好构造高斯尺度空间的第一张图像，该图像对应的尺度为1.6.
  图像由原始输入图像经2倍上采样得到，并经过处理后，使得它对应的高斯尺度为1.6
  img，原始输入图像，我们要对该图像检测尺度不变特征点
  sigma，高斯尺度空间第一张图像所对一会给的尺度；根据Lowe建议，该值为1.6
*/
static IplImage* create_init_img( IplImage* img, double sigma )
{
  IplImage* gray, * dbl; //gray是原始图像转换之后的灰度图像，dbl是2倍上采样图像
  double sig_diff;

  gray = convert_to_gray32( img ); //高斯尺度空间只能对灰度图像建立，所以先转成灰度图像

  //按照Lowe建议，把原始输入图像进行2倍上采样得到dbl。这样构造出的高斯尺度空间的第一层的图像dbl实际上是输入图像通过线性插值上采样2倍的结果
  //并且Lowe建议，对dbl进行尺度为sigma = 1.6的滤波；Lowe假设原始输入图像img经过相机镜头拍摄时实际上已经经过了高斯滤波，假设尺度为0.5，即
  //SIFT_INIT_SIGMA=0.5,那放大2倍时dbl的“自带”尺度相应的就成了2*SIFT_INIT_SIGMA，所以为了使滤波后的dbl的尺度为1.6,我们对其施加的高斯滤波的尺度
  //只需要为sig_diff = sqrt(1.6^2-(SIFT_INIT_SIGMA*2)^2)
  sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4 );
  dbl = cvCreateImage( cvSize( img->width*2, img->height*2 ), IPL_DEPTH_32F, 1 ); 
  cvResize( gray, dbl, CV_INTER_CUBIC ); //2倍上采样，采用了双三次插值
  cvSmooth( dbl, dbl, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff ); //对dbl进行高斯滤波，滤波之后的dbl对应的高斯尺度为sigma=1.6
  cvReleaseImage( &gray );

  return dbl;
}

/*
 把输入图像转换成32bit灰度图，就是转换成灰度图且把取值范围归一化到[0,1]之内，pixelValue/255.0

 img，输入图像，可以是8-bit灰度图，也可以是BGR彩色图
*/
static IplImage* convert_to_gray32( IplImage* img )
{
  IplImage* gray8, * gray32;
  
  gray32 = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );
  if( img->nChannels == 1 )
    gray8 = cvClone( img );
  else
    {
      gray8 = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
      cvCvtColor( img, gray8, CV_BGR2GRAY );
    }
  cvConvertScale( gray8, gray32, 1.0 / 255.0, 0 );

  cvReleaseImage( &gray8 );
  return gray32;
}

/*
  构建高斯尺度空间金字塔

  base，高斯尺度空间的初始层，对应于尺度1.6
  octvs，组数
  intvls，每组分隔数，每组层数为intvls+3
  sigma, 每组第0层的高斯卷积核，要注意每组对应层的高斯卷积核实际上是一样大的，相邻组对应层2倍的尺度关系通过下采样来体现

  返回gauss_pyr，高斯尺度空间金字塔，它的总层数为octvs x (intvls + 3)
*/
static IplImage*** build_gauss_pyr( IplImage* base, int octvs, int intvls, double sigma )
{
  IplImage*** gauss_pyr; //高斯金字塔
  double *sig = (double*)calloc(intvls+3, sizeof(double));

  gauss_pyr = calloc( octvs, sizeof( IplImage** ) );
  for( int i = 0; i < octvs; i++ )
    gauss_pyr[i] = calloc( intvls + 3, sizeof( IplImage *) );

  /*
    预先计算好每组使用的高斯卷积核的标准差
	需要注意：每组的高斯标准差都是一样的，这是因为相邻两组对应层的尺度为2倍关系，但相邻两组的空间分辨率正好减半
	所以高斯尺度上放大两倍的关系就转化为图像空间分辨率减半上了，因此每组对应层实际卷积用的高斯标准差是一样的。但不要理解混淆，
	相邻两组对应层之间的高斯尺度关系为2倍关系

	在有第i层得到第k+1层时，只需要对第i层进行标准差为sqrt(sigma_{i+1}^2 - sigma_{i}^2)的高斯滤波
    sig[i+1]存储的是从第i层生成i+1层时，需要对第i层进行的高斯卷积的标准差（i>0）。
	当然，每组第0层是不用进行卷积的：第一组的第0层是事先准备好的，尺度为1.6;
	以后每一组的第0层由上一组倒数第三层结果进行1/2下采样得到，其高斯尺度与上一组倒数第三层一样
  */
  double k = pow( 2.0, 1.0 / intvls ); //高斯尺度空间中，相邻两层之间尺度的倍数为k
  sig[0] = sigma; //每组第0层尺度
  sig[1] = sigma * sqrt( k*k- 1 ); //每组第1层尺度增量
  for (int i = 2; i < intvls + 3; i++)
      sig[i] = sig[i-1] * k; //从第2层开始，每层尺度增量正好是上一个增量乘k

  for (int o = 0; o < octvs; o++) //遍历组
  {
	  for (int i = 0; i < intvls + 3; i++) //组内遍历层，每组是intvls + 3层
	  {
		  if (o == 0 && i == 0) //高斯尺度空间初始层
			  gauss_pyr[o][i] = cvCloneImage(base);
		  else if (i == 0)/* 开始新的一组了，第0层为上一组倒数第三层进行1/2下采样 */
			  gauss_pyr[o][i] = downsample(gauss_pyr[o - 1][intvls]);
		  else/*在本组内，第i层由对第i-1层施加标准差为sig[i]的高斯卷积得到*/
		  {
			  gauss_pyr[o][i] = cvCreateImage(cvGetSize(gauss_pyr[o][i - 1]), IPL_DEPTH_32F, 1);
			  cvSmooth(gauss_pyr[o][i - 1], gauss_pyr[o][i], CV_GAUSSIAN, 0, 0, sig[i], sig[i]);
		  }
	  }
  }
  free(sig);
  return gauss_pyr;
}

/*
  对图像进行1/2降采样
*/
static IplImage* downsample( IplImage* img )
{
  IplImage* smaller = cvCreateImage( cvSize(img->width / 2, img->height / 2), img->depth, img->nChannels );
  cvResize( img, smaller, CV_INTER_LINEAR);

  return smaller;
}

/*
  //从高斯尺度空间金字塔构建处DoG尺度空间金字塔
  gauss_pyr，高斯尺度空间金字塔
  octvs，高斯尺度空间金字塔的组数
  intvls，高斯尺度空间金字塔每组间隔数，每组层数为intvls+3

  返回dog_pyr，为DOG尺度空间，其层数为 octvs * (intvls + 2) 
*/
static IplImage*** build_dog_pyr( IplImage*** gauss_pyr, int octvs, int intvls )
{
  IplImage*** dog_pyr; //DoG尺度空间金字塔，每组有intvls+2层

  dog_pyr = calloc( octvs, sizeof( IplImage** ) );
  for(int i = 0; i < octvs; i++ )
    dog_pyr[i] = calloc( intvls + 2, sizeof(IplImage*) );

  for (int o = 0; o < octvs; o++)
  {
	  for (int i = 0; i < intvls + 2; i++)
	  {
		  dog_pyr[o][i] = cvCreateImage(cvGetSize(gauss_pyr[o][i]), IPL_DEPTH_32F, 1);
		  //DoG空间里面的层由高斯尺度空间中对应的相邻两层相减得到
		  cvSub(gauss_pyr[o][i + 1], gauss_pyr[o][i], dog_pyr[o][i], NULL);
	  }
  }
  return dog_pyr;
}

/*
  在DoG尺度空间金字塔中寻找尺度不变特征点
  dog_pyr，DoG尺度空间金字塔
  octvs，DOG尺度空间中的组数
  intvls，每组间隔数，每组层数为intvls+2
  contr_thr,如果DoG中点的响应值小于该阈值，则该点不是特征点
  @param curv_thr high threshold on feature ratio of principal curvatures
  @param storage memory storage in which to store detected features

  返回尺度不变特征点集合
*/
static CvSeq* scale_space_extrema( IplImage*** dog_pyr, int octvs, int intvls, double contr_thr, int curv_thr, CvMemStorage* storage )
{
  CvSeq* features;
  double prelim_contr_thr = 0.5 * contr_thr / intvls;
  struct feature* feat;
  struct detection_data* ddata;
  unsigned long* feature_mat;

  features = cvCreateSeq( 0, sizeof(CvSeq), sizeof(struct feature), storage );
  for( int octaveIndex = 0; octaveIndex < octvs; octaveIndex++ )
  {
    feature_mat = calloc( dog_pyr[octaveIndex][0]->height * dog_pyr[octaveIndex][0]->width, sizeof(unsigned long) );
	for (int layerIndex = 1; layerIndex <= intvls; layerIndex++) //对layerIndex所在的层进行潜在特征点检测，该层要与其下与其上的层进行比较
	{
		for (int r = SIFT_IMG_BORDER; r < dog_pyr[octaveIndex][0]->height - SIFT_IMG_BORDER; r++) //r为行索引
		{
			for (int c = SIFT_IMG_BORDER; c < dog_pyr[octaveIndex][0]->width - SIFT_IMG_BORDER; c++)//c为列索引
			{
				/* 该点的响应值的绝对值需要大于一个阈值prelim_contr_thr*/
				if (ABS(pixval32f(dog_pyr[octaveIndex][layerIndex], r, c)) > prelim_contr_thr)
				{
					if (is_extremum(dog_pyr, octaveIndex, layerIndex, r, c)) //如果[o,layerIndex,r,c]点为一个局部极值点
					{
						//对[octaveIndex, layerIndex, r, c]所确定的整数位置上的极值点进行位置精化
						feat = interp_extremum(dog_pyr, octaveIndex, layerIndex, r, c, intvls, contr_thr);
						if (feat) //精化操作成功
						{
							ddata = feat_detection_data(feat);
							//判断该特征点是否是委员图像edge之上，如果是的话，该点不被认为是特征点
							//注意，是不是边缘点的判断是在初始离散位置上判定的，而不是在位置精化之后的点上进行的，因为位置精化之后的点
							//在尺度空间中的坐标都不是整数了，再求它的Hessian矩阵就非常困难了
							if (!is_too_edge_like(dog_pyr[ddata->octv][ddata->intvl], ddata->r, ddata->c, curv_thr))
								cvSeqPush(features, feat);
							else
								free(ddata);
							free(feat);
						}
					}
				}
			}
		}
	}
    free( feature_mat );
  }
  return features;
}

/*
  判断在DoG尺度空间金字塔中，在组octv，在层intvl，点[r,c]的值是否为极值点,该点需要与它在尺度空间中的26个邻居进行比较

  dog_pyr，DoG尺度空间金字塔
  octv，待判断的点所在的组的索引
  layer，待判断的点所在组的层的索引
  r，待判断的点的行号
  c，待判断的点的列号

  返回1，则说明该点为极值点，返回0说明不是极值点
*/
static int is_extremum( IplImage*** dog_pyr, int octvIndex, int layerIndex, int r, int c )
{
  double val = pixval32f( dog_pyr[octvIndex][layerIndex], r, c ); //返回该点在DoG尺度空间中的值

  /*DoG滤波器为零均值滤波器。若该点响应值大于0，我们需要判断它是否为极大值 */
  if( val > 0 )
    {
	  //与它的26个尺度空间中的邻居进行比对，只要有比它大的，就说明它肯定不是极值点
	  for (int i = -1; i <= 1; i++)
	  {
		  for (int j = -1; j <= 1; j++)
		  {
			  for (int k = -1; k <= 1; k++)
			  {
				  if (val < pixval32f(dog_pyr[octvIndex][layerIndex + i], r + j, c + k))
					  return 0;
			  }
		  }
	  }
    }
  else/*判断它是否为局部极小值 */
    {
	  for (int i = -1; i <= 1; i++)
	  {
		  for (int j = -1; j <= 1; j++)
		  {
			  for (int k = -1; k <= 1; k++)
			  {
				  if (val > pixval32f(dog_pyr[octvIndex][layerIndex + i], r + j, c + k))
					  return 0;
			  }
		  }
	  }
    }

  return 1;
}

/*
  对DoG中初始得到的局部极值点的位置进行精化，以得到更加准确的极值点位置（空间位置以及尺度位置）
  基于的原理就是对该点局部尺度空间信息进行二阶泰勒展开，然后求局部极小值点，此过程最多迭代进行5次

  @param dog_pyr，DoG尺度空间金字塔
  @param octvIndex，当前初始极值点在DoG尺度空间金字塔中的组索引
  @param layerIndex，当前这个初始极值点在组octv中的层索引
  @param r feature's image row
  @param c feature's image column
  @param intvls，DoG尺度空间金字塔的每组内的间隔数
  @param contr_thr,如果该点的对比度低于阈值contr_thr，则也不认为该点为特征点

  @return Returns the feature resulting from interpolation of the given
    parameters or NULL if the given location could not be interpolated or
    if contrast at the interpolated loation was too low.  If a feature is
    returned, its scale, orientation, and descriptor are yet to be determined.
*/
static struct feature* interp_extremum( IplImage*** dog_pyr, int octvIndex, int layerIndex, int r, int c, int intvls,double contr_thr )
{
  struct feature* feat;
  struct detection_data* ddata;
  double xi, xr, xc, contr;
  int iterationTimes = 0;
  
  while(iterationTimes < SIFT_MAX_INTERP_STEPS ) //最多尝试5次
    {
	  //对当前整数极值点位置[octv, layerIndex, r, c]进行二次插值，得到更新量 （xi, xr, xc）
      interp_step( dog_pyr, octvIndex, layerIndex, r, c, &xi, &xr, &xc );
      if( ABS( xi ) < 0.5  &&  ABS( xr ) < 0.5  &&  ABS( xc ) < 0.5 ) //如果沿着尺度空间任意维度的更新量都小于0.5，就说明不用更新了
		break;
      
	  //更新到新的DOG尺度空间中整数位置点
      c += cvRound( xc );
      r += cvRound( xr );
	  layerIndex += cvRound( xi ); //注意，Lowe论文里面说，对尺度sigma求导，这是不准确的，实际上是对尺度所对应的层求导，相邻层是相连的间隔为1的整数
	 
      //对更新之后的新的整数位置点进行一些检查，不满足以下条件的话（比如超出了本组层数的边界），就认为更新失败
      if(layerIndex < 1  || layerIndex > intvls  || c < SIFT_IMG_BORDER  || r < SIFT_IMG_BORDER  
		  || c >= dog_pyr[octvIndex][0]->width - SIFT_IMG_BORDER  || r >= dog_pyr[octvIndex][0]->height - SIFT_IMG_BORDER )
	  {
		  return NULL;
	  }
	  iterationTimes++;
    }
  
  /* ensure convergence of interpolation */
  //这个判断很关键，它意味着上面的while循环中的break一定要触发，如果不触发，就认为没有收敛
  //break触发了，说明最后一次算出来的xr和xc还没有更新到r,c,和layerIndex之上
  if(iterationTimes >= SIFT_MAX_INTERP_STEPS )
    return NULL;
  
  //计算在插值点处的DoG响应值，如果此值很小，则该点不被认为是特征点
  contr = interp_contr( dog_pyr, octvIndex, layerIndex, r, c, xi, xr, xc );
  if( ABS( contr ) < contr_thr / intvls )
    return NULL;

  feat = new_feature();
  ddata = feat_detection_data( feat );
  feat->img_pt.x = feat->x = ( c + xc ) * pow( 2.0, octvIndex); //从下采样组的空间分辨率返回到原始图像2倍大小的空间位置
  feat->img_pt.y = feat->y = ( r + xr ) * pow( 2.0, octvIndex);
  //ddata是特征点feat在尺度空间中的信息，在那一层的行、列，在哪一组、哪一层、亚层偏移量
  //注意：ddata中的r,c,layerIndex是最后一次精化操作之前的在DoG尺度空间中的位置信息，都为整数。也就是说
  //最后一次偏移量估计是基于该点[r,c,layerIndex]进行泰勒展开的。之所以要记录这个整数位置，是为了后面判断边缘点和构建描述子方便
  ddata->r = r; 
  ddata->c = c;
  ddata->octv = octvIndex;
  ddata->intvl = layerIndex;
  ddata->subintvl = xi; //记录下在亚层的偏移量

  return feat;
}

/*
  Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's
  paper.
  从当前整数极值点[octv, intvl, r, c]计算极值点位置对应的更新量
  @param dog_pyr，DoG尺度空间金字塔
  @param octvIndex，组索引 octave of scale space
  @param layerIndex，组内层索引
  @param r，行索引 
  @param c，列索引
  @param xi，对层的更新量
  @param xr，对行的更新量
  @param xc，对列的更新量
*/

static void interp_step( IplImage*** dog_pyr, int octvIndex, int layerIndex, int r, int c, double* xi, double* xr, double* xc )
{
  CvMat* dD, * H, * H_inv, X;
  double x[3] = { 0 };
  
  //返回一个三维向量，DoG(x, y, sigma)在点[octvIndex, layerIndex, r, c]处{ dDoG / dx, dDoG / dy, dDog / dl }^T
  dD = deriv_3D( dog_pyr, octvIndex, layerIndex, r, c );

  //返回DoG尺度空间在[octvIndex, layerIndex, r, c]处的海森矩阵
  H = hessian_3D( dog_pyr, octvIndex, layerIndex, r, c );
  H_inv = cvCreateMat( 3, 3, CV_64FC1 ); //H_inv为海森矩阵的逆矩阵
  cvInvert( H, H_inv, CV_SVD );
  cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
  cvGEMM( H_inv, dD, -1, NULL, 0, &X, 0 ); //- H^(-1)*Gradient,更新增量放在了X中
  
  cvReleaseMat( &dD );
  cvReleaseMat( &H );
  cvReleaseMat( &H_inv );

  *xi = x[2];
  *xr = x[1];
  *xc = x[0];
}

/*
  在DoG尺度空间中，计算DoG(x,y,l)在点（x,y,l）处的偏导数,l为层索引
  @param dog_pyr，DoG尺度空间金字塔
  @param octvIndex， 点的组索引
  @param layerIndex，点的组内层索引
  @param r，点的行索引
  @param c，点的列索引

  返回一个三维向量，DoG(x,y,sigma)在点[octvIndex, layerIndex, r, c]处{ dDoG/dx, dDoG/dy, dDog/l }^T 
*/
static CvMat* deriv_3D( IplImage*** dog_pyr, int octvIndex, int layerIndex, int r, int c )
{
  CvMat* dI;
  double dx, dy, ds;

  dx = ( pixval32f( dog_pyr[octvIndex][layerIndex], r, c+1 ) - pixval32f( dog_pyr[octvIndex][layerIndex], r, c-1 ) ) / 2.0;
  dy = ( pixval32f( dog_pyr[octvIndex][layerIndex], r+1, c ) - pixval32f( dog_pyr[octvIndex][layerIndex], r-1, c ) ) / 2.0;
  ds = ( pixval32f( dog_pyr[octvIndex][layerIndex +1], r, c ) - pixval32f( dog_pyr[octvIndex][layerIndex -1], r, c ) ) / 2.0;
  
  dI = cvCreateMat( 3, 1, CV_64FC1 );
  cvmSet( dI, 0, 0, dx );
  cvmSet( dI, 1, 0, dy );
  cvmSet( dI, 2, 0, ds );

  return dI;
}

/*
  计算DoG(x,y,sigma)的海森矩阵
  dog_pyr，DoG尺度空间
  @param octvIndex,组索引
  @param layerIndex，层索引
  @param r，行索引
  @param c，列索引

  @返回点[octvIndex, layerIndex, r, c]处DoG(x,y,layerIndex)的海森矩阵

  / DOGxx  DOGxy  DOGxs \ <BR>
  | DOGxy  DOGyy  DOGys | <BR>
  \ DOGxs  DOGys  DOGss /
*/
static CvMat* hessian_3D( IplImage*** dog_pyr, int octvIndex, int layerIndex, int r, int c )
{
  CvMat* H;
  double v, dxx, dyy, dss, dxy, dxs, dys;
  
  v = pixval32f( dog_pyr[octvIndex][layerIndex], r, c );
  dxx = ( pixval32f( dog_pyr[octvIndex][layerIndex], r, c+1 ) + pixval32f( dog_pyr[octvIndex][layerIndex], r, c-1 ) - 2 * v );
  dyy = ( pixval32f( dog_pyr[octvIndex][layerIndex], r+1, c ) + pixval32f( dog_pyr[octvIndex][layerIndex], r-1, c ) - 2 * v );
  dss = ( pixval32f( dog_pyr[octvIndex][layerIndex +1], r, c ) + pixval32f( dog_pyr[octvIndex][layerIndex -1], r, c ) - 2 * v );
  dxy = ( pixval32f( dog_pyr[octvIndex][layerIndex], r+1, c+1 ) -
	  pixval32f( dog_pyr[octvIndex][layerIndex], r+1, c-1 ) -
	  pixval32f( dog_pyr[octvIndex][layerIndex], r-1, c+1 ) +
	  pixval32f( dog_pyr[octvIndex][layerIndex], r-1, c-1 ) ) / 4.0;
  dxs = ( pixval32f( dog_pyr[octvIndex][layerIndex +1], r, c+1 ) -
	  pixval32f( dog_pyr[octvIndex][layerIndex +1], r, c-1 ) -
	  pixval32f( dog_pyr[octvIndex][layerIndex -1], r, c+1 ) +
	  pixval32f( dog_pyr[octvIndex][layerIndex -1], r, c-1 ) ) / 4.0;
  dys = ( pixval32f( dog_pyr[octvIndex][layerIndex +1], r+1, c ) -
	  pixval32f( dog_pyr[octvIndex][layerIndex +1], r-1, c ) -
	  pixval32f( dog_pyr[octvIndex][layerIndex -1], r+1, c ) +
	  pixval32f( dog_pyr[octvIndex][layerIndex -1], r-1, c ) ) / 4.0;
  
  H = cvCreateMat( 3, 3, CV_64FC1 );
  cvmSet( H, 0, 0, dxx );
  cvmSet( H, 0, 1, dxy );
  cvmSet( H, 0, 2, dxs );
  cvmSet( H, 1, 0, dxy );
  cvmSet( H, 1, 1, dyy );
  cvmSet( H, 1, 2, dys );
  cvmSet( H, 2, 0, dxs );
  cvmSet( H, 2, 1, dys );
  cvmSet( H, 2, 2, dss );

  return H;
}

/*
  计算极值点处的DoG响应值，注意这个极值点位置是经过精化处理之后的
  @param dog_pyr，DoG尺度空间金字塔
  @param octv,组索引
  @param layerIndex，原始整数极值点处的层索引
  @param r，原始整数极值点处的行索引
  @param c，原始整数极值点处的列索引
  @param xi，层偏移量
  @param xr，行偏移量
  @param xc，列偏移量

  返回插值点处的DoG响应值
*/
static double interp_contr( IplImage*** dog_pyr, int octvIndex, int layerIndex, int r, int c, double xi, double xr, double xc )
{
  CvMat* dD, X, T;
  double t[1], x[3] = { xc, xr, xi };

  //实现教材公式4-16
  cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
  cvInitMatHeader( &T, 1, 1, CV_64FC1, t, CV_AUTOSTEP );
  dD = deriv_3D( dog_pyr, octvIndex, layerIndex, r, c );
  cvGEMM( dD, &X, 1, NULL, 0, &T,  CV_GEMM_A_T );
  cvReleaseMat( &dD );

  return pixval32f( dog_pyr[octvIndex][layerIndex], r, c ) + t[0] * 0.5;
}

/*
  构造并初始化一个SIFT特征，包括了特征点的位置信息和特征描述子
*/
static struct feature* new_feature( void )
{
  struct feature* feat;
  struct detection_data* ddata;

  feat = malloc( sizeof( struct feature ) );
  memset( feat, 0, sizeof( struct feature ) );
  ddata = malloc( sizeof( struct detection_data ) );
  memset( ddata, 0, sizeof( struct detection_data ) );
  feat->feature_data = ddata;
  feat->type = FEATURE_LOWE;

  return feat;
}

/*
  判断一个精化后特征点是否位于图像边缘之上。注意：待判断的点实际上是离精化后特征点最近位置的DoG尺度空间中整数位置处的点

  @param dog_img，DoG尺度空间中的一层，在该层的[r,c]位置为待判断的特征点
  @param r，特征点在DoG该层中的行
  @param c，特征点在DoG该层中的列
  @param curv_thr，阈值，该值按照David Lowe建议设置为10
*/
static int is_too_edge_like( IplImage* dog_img, int r, int c, int curv_thr )
{
  double d, dxx, dyy, dxy, tr, det;

  /* 计算该点的Hessian矩阵中的元素 */
  d = pixval32f(dog_img, r, c);
  dxx = pixval32f( dog_img, r, c+1 ) + pixval32f( dog_img, r, c-1 ) - 2 * d;
  dyy = pixval32f( dog_img, r+1, c ) + pixval32f( dog_img, r-1, c ) - 2 * d;
  dxy = ( pixval32f(dog_img, r+1, c+1) - pixval32f(dog_img, r+1, c-1) - pixval32f(dog_img, r-1, c+1) + pixval32f(dog_img, r-1, c-1) ) / 4.0;
  tr = dxx + dyy; //Hessian矩阵的迹
  det = dxx * dyy - dxy * dxy; //Hessian矩阵的行列式

  /*如果行列式小于0，说明该点的两个主曲率异号，则该点不是局部极值点，舍弃 */
  if( det <= 0 )
    return 1;

  if( tr * tr / det < ( curv_thr + 1.0 )*( curv_thr + 1.0 ) / curv_thr )
    return 0;
  return 1;
}

/*
  计算特征点的尺度，由于我们之前已经知道了它所在的层以及层的偏移量，可以插值计算出精确的尺度值
  @param features array of features
  @param sigma,高斯尺度空间的初始尺度，也是每一组的初始尺度，注意：每一组的初始尺度sigma是一样的，2倍关系是通过分辨率减半来体现的
  @param intvls intervals per octave of scale space
*/
static void calc_feature_scales( CvSeq* features, double sigma, int intvls )
{
  struct feature* feat;
  struct detection_data* ddata;
  double intvl;

  for(int featureIndex = 0; featureIndex < features->total; featureIndex++ )
    {
      feat = CV_GET_SEQ_ELEM( struct feature, features, featureIndex);
      ddata = feat_detection_data( feat );
      intvl = ddata->intvl + ddata->subintvl; //精确到尺度空间亚层
	  //计算该特征点的特征尺度
      feat->scl = sigma * pow( 2.0, ddata->octv + intvl / intvls );
	  //ddata->scl_octv，是该特征点相对于这一组初始层的尺度，但它也是精确到亚层的尺度
      ddata->scl_octv = sigma * pow( 2.0, intvl / intvls );
    }
}

/*
  因为在构建尺度空间时，初始图像的分辨率提升至了输入图像的2倍，而我们最终的特征点位置是以输入图像为基准的
*/
static void adjust_for_img_dbl( CvSeq* features )
{
  struct feature* feat;
  for( int i = 0; i < features->total; i++ )
    {
      feat = CV_GET_SEQ_ELEM( struct feature, features, i );
      feat->x /= 2.0;
      feat->y /= 2.0;
      feat->scl /= 2.0;
      feat->img_pt.x /= 2.0;
      feat->img_pt.y /= 2.0;
    }
}

/*这一步要计算特征点的主方向，如果某个特征点有多于一个的主方向的话，需要复制几份，每个都当作独立的特征点进行后续处理。
也就是说，在某个位置、某个特征尺度上，有可能存在几个具有不同主方向的特征点
  @features，一个数组，每一个元素是一个特征
  @gauss_pyr，高斯尺度空间
*/
static void calc_feature_oris( CvSeq* features, IplImage*** gauss_pyr )
{
  struct feature* feat;
  struct detection_data* ddata;
  double* hist;
  double omax;

  for( int i = 0; i < features->total; i++ )
  {
      feat = malloc( sizeof( struct feature ) );
      cvSeqPopFront( features, feat ); //这个操作是把features数组当中最上面的一个删除，并把它临时存储在feat中
      ddata = feat_detection_data( feat );
	  //特征点主方向的判断要根据梯度方向直方图来判断，这个梯度方向直方图要在与特征尺度对应的高斯层上进行
      hist = ori_hist(gauss_pyr[ddata->octv][ddata->intvl], ddata->r, ddata->c, SIFT_ORI_HIST_BINS, cvRound( SIFT_ORI_RADIUS * ddata->scl_octv ), SIFT_ORI_SIG_FCTR * ddata->scl_octv );
	  for (int j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++)
	  {
		  //对直方图进行2次平滑
		  smooth_ori_hist(hist, SIFT_ORI_HIST_BINS);
	  }
      omax = dominant_ori(hist, SIFT_ORI_HIST_BINS ); //找到梯度方向直方图hist中最大峰值
      add_good_ori_features(features, hist, SIFT_ORI_HIST_BINS, omax * SIFT_ORI_PEAK_RATIO, feat);
      free( ddata );
      free( feat );
      free( hist );
  }
}

/*
  在图像img上一点计算梯度方向直方图
  img，输入图像
  r，像素所在行
  c，像素所在列
  n，方向直方图的柱数目
  rad，计算直方图时使用的区域半径, 4.5*characterisitc scale
  sigma, 在计算梯度直方图时，每点的贡献需要按照该点到中心距离的高斯加权，该高斯函数的std为sigma= 1.5*characterisitc scale

  返回一个数组，表示梯度方向直方图，角度范围为0~2PI
*/
static double* ori_hist( IplImage* img, int r, int c, int n, int rad, double sigma )
{
  double* hist;
  double mag, ori, w, exp_denom, PI2 = CV_PI * 2.0;
  int bin;

  hist = calloc( n, sizeof( double ) );
  exp_denom = 2.0 * sigma * sigma;
  for (int i = -rad; i <= rad; i++)
  {
	  for (int j = -rad; j <= rad; j++)
	  {
		  if (calc_grad_mag_ori(img, r + i, c + j, &mag, &ori)) //计算该点梯度幅值和角度
		  {
			  w = exp(-(i*i + j * j) / exp_denom); //该点(i,j)对梯度直方图的贡献要按照其距离中心点的距离高斯加权
			  //要注意，ori的取值范围为[-PI, PI], 所以需要加上CV_PI，将其范围调整至[0, 2PI]
			  //先把角度范围转换到(0,1)之间，然后乘上n，看看它落在哪个bin里面
			  bin = cvRound(n * (ori + CV_PI) / PI2); 
			  bin = (bin < n) ? bin : 0;
			  hist[bin] += w * mag; //加权更新对应的bin值
		  }
	  }
  }
  return hist;
}

/*
  计算图像上一点的梯度幅值与角度，角度的范围为[-PI, PI]
  @param img image
  @param r pixel row
  @param c pixel col
  @param mag output as gradient magnitude at pixel (r,c)
  @param ori output as gradient orientation at pixel (r,c),其取值范围为[-PI, PI]

  @return Returns 1 if the specified pixel is a valid one and sets mag and
    ori accordingly; otherwise returns 0
*/
static int calc_grad_mag_ori( IplImage* img, int r, int c, double* mag, double* ori )
{
  double dx, dy;

  if( r > 0  &&  r < img->height - 1  &&  c > 0  &&  c < img->width - 1 ) //进行边界测试，太靠近图像边界的点就不考虑了
    {
      dx = pixval32f( img, r, c+1 ) - pixval32f( img, r, c-1 );
      dy = pixval32f( img, r-1, c ) - pixval32f( img, r+1, c );
      *mag = sqrt( dx*dx + dy*dy ); //梯度的幅值
      *ori = atan2( dy, dx ); //弧度范围是[-PI, PI]
      return 1;
    }
  else
    return 0;
}


/*
  对梯度方向直方图进行高斯平滑
  hist，梯度方向直方图
  n，直方图的bin的数目
*/
static void smooth_ori_hist( double* hist, int n )
{
  double prev, tmp, h0 = hist[0];
  int i;

  prev = hist[n-1];
  for( i = 0; i < n; i++ )
    {
      tmp = hist[i];
      hist[i] = 0.25 * prev + 0.5 * hist[i] + 0.25 * ( ( i+1 == n )? h0 : hist[i+1] );
      prev = tmp;
    }
}

/*
  寻找梯度方向直方图中的最大峰值
  hist， an orientation histogram
  n， number of bins

  返回梯度方向直方图hist中的最大峰值
*/
static double dominant_ori( double* hist, int n )
{
  double omax;
  int maxbin;

  omax = hist[0];
  maxbin = 0;
  for (int i = 1; i < n; i++)
  {
	  if (hist[i] > omax)
	  {
		  omax = hist[i];
		  maxbin = i;
	  }
  }
  return omax;
}



/*
  Interpolates a histogram peak from left, center, and right values
*/
#define interp_hist_peak( l, c, r ) ( 0.5 * ((l)-(r)) / ((l) - 2.0*(c) + (r)) )



/*
  如果该点的梯度方向直方图有多个主方向（对应的bin的值超过最大值的80%），则对每一个主方向都复制一个特征点出来
  也就是说，在该点处，在同样的特征尺度下，有几个不同的特征点，它们就是主方向不同。这样可以增加匹配的稳定性。

  features，特征集合
  hist，当前该点的梯度方向直方图
  n，直方图的bin的数目
  mag_thr，一个阈值，如果直方图某个bin的值超过该值，则复制一份特征
  feat，当前这个特征点已经有了一个feature对象；如果该点有多个主方向的话，就要复制多份feature,就从feat复制的
*/
static void add_good_ori_features( CvSeq* features, double* hist, int n, double mag_thr, struct feature* feat )
{
  struct feature* new_feat;
  double bin, PI2 = CV_PI * 2.0;
  int l, r;

  for( int binIndex = 0; binIndex < n; binIndex++ )
    {
      l = (binIndex == 0 )? n - 1 : binIndex -1; //当前binIndex的左边的binIndex；若当前binIndex为0，其左边binIndex为n-1
      r = (binIndex + 1 ) % n; //当前binIndex右边的binIndex；若当前binIndex为n-1，其右边binIndex为0
      
	  //当前binIndex所在之处为hist中的一个局部峰值，且大于阈值mag_thr
	  //我们需要对它进行方向插值，并把feat复制一份，生成多个特征点
      if( hist[binIndex] > hist[l]  &&  hist[binIndex] > hist[r]  &&  hist[binIndex] >= mag_thr )
	  {
		  bin = binIndex + interp_hist_peak(hist[l], hist[binIndex], hist[r] );
		  bin = ( bin < 0 )? n + bin : ( bin >= n )? bin - n : bin;
		  new_feat = clone_feature( feat );
		  new_feat->ori = ( ( PI2 * bin ) / n ) - CV_PI; //最终的主方向范围在[-PI, PI]之间
		  //new_feat->ori = ((PI2 * bin) / n) ;
		  cvSeqPush( features, new_feat );
		  free( new_feat );
	  }
    }
}

/*
  Makes a deep copy of a feature

  @param feat feature to be cloned

  @return Returns a deep copy of feat
*/
static struct feature* clone_feature( struct feature* feat )
{
  struct feature* new_feat;
  struct detection_data* ddata;

  new_feat = new_feature();
  ddata = feat_detection_data( new_feat );
  memcpy( new_feat, feat, sizeof( struct feature ) );
  memcpy( ddata, feat_detection_data(feat), sizeof( struct detection_data ) );
  new_feat->feature_data = ddata;

  return new_feat;
}

/*
  计算SIFT特征描述子
  features数组当中的每个元素存储了特征点的位置，特征尺度，特征尺度所在组和层，还有主方向信息
  descr_width值为4，descr_hist_bins值为8，这样SIFT描述子的维度就是4*4*8=128
  gauss_pyr，高斯尺度空间，SIFT的计算是要在离特征点最近的高斯尺度层上来进行的
*/
static void compute_descriptors( CvSeq* features, IplImage*** gauss_pyr, int descr_width, int descr_hist_bins)
{
  struct feature* feat;
  struct detection_data* ddata;
  double*** hist;

  for( int featIndex = 0; featIndex < features->total; featIndex++ )
    {
      feat = CV_GET_SEQ_ELEM( struct feature, features, featIndex);
      ddata = feat_detection_data( feat );
      hist = descr_hist(gauss_pyr[ddata->octv][ddata->intvl], ddata->r, ddata->c, feat->ori, ddata->scl_octv, descr_width, descr_hist_bins);
      hist_to_descr( hist, descr_width, descr_hist_bins, feat );
      release_descr_hist( &hist, descr_width);
    }
}



/*
  计算SIFT特征描述子

  img，高斯尺度空间中的对应于该特征点的一层
  r，c,在该高斯层上特征点的位置
  ori，特征点的主方向
  scl,组内该高斯尺度层的相对于该组第0层的尺度
  descr_width， width of 2d array of orientation histograms
  descr_hist_bins， bins per orientation histogram

  返回一个  descr_width x descr_width数组，每个元素为 n-bin的梯度方向直方图
*/
static double*** descr_hist( IplImage* img, int r, int c, double ori, double scl, int descr_width, int descr_hist_bins)
{
  double*** hist;
  double cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot, grad_mag, grad_ori, w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * CV_PI;
  int radius;

  hist = calloc(descr_width, sizeof( double** ) );
  //开辟直方图数组空间，4*4个直方图，每个直方图8个bin
  for( int i = 0; i < descr_width; i++ )
  {
      hist[i] = calloc(descr_width, sizeof( double* ) );
	  for (int j = 0; j < descr_width; j++)
	  {
		  hist[i][j] = calloc(descr_hist_bins, sizeof(double));
	  }
  }
  
  cos_t = cos( ori );
  sin_t = sin( ori );
  bins_per_rad = descr_hist_bins / PI2;
  exp_denom = descr_width * descr_width * 0.5;
  hist_width = SIFT_DESCR_SCL_FCTR * scl; //3sigma原则确定用于计算描述子的邻域半径
  radius = hist_width * sqrt(2) * (descr_width + 1.0 ) * 0.5 + 0.5;
  //radius = hist_width * sqrt(2) * (descr_width) * 0.5 + 0.5;
  for (int i = -radius; i <= radius; i++)
  {
	  for (int j = -radius; j <= radius; j++)
	  {
		  /*
			Calculate sample's histogram array coords rotated relative to ori.
			Subtract 0.5 so samples that fall e.g. in the center of row 1 (i.e.
			r_rot = 1.5) have full weight placed in row 1 after interpolation.
		  */
		  c_rot = (j * cos_t - i * sin_t) / hist_width;
		  r_rot = (j * sin_t + i * cos_t) / hist_width;
		  rbin = r_rot + descr_width / 2 - 0.5; //确定当前这一点应该属于hist哪一个bin，hist是4*4的二维数组指针，每个元素是一个指向
		  //8维向量的指针
		  cbin = c_rot + descr_width / 2 - 0.5;

		  if (rbin > -1.0  &&  rbin < descr_width  &&  cbin > -1.0  &&  cbin < descr_width)
		  {
			  if (calc_grad_mag_ori(img, r + i, c + j, &grad_mag, &grad_ori))
			  {
				  grad_ori -= ori; //方向归一化
				  while (grad_ori < 0.0) //方向值要统一到[0,2PI]
					  grad_ori += PI2;
				  while (grad_ori >= PI2)
					  grad_ori -= PI2;

				  obin = grad_ori * bins_per_rad; //找到当前这个点的梯度方向应该落在方向直方图哪个bin里面
				  w = exp(-(c_rot * c_rot + r_rot * r_rot) / exp_denom);
				  interp_hist_entry(hist, rbin, cbin, obin, grad_mag * w, descr_width, descr_hist_bins);
			  }
		  }
	  }
  }
  return hist;
}

/*
  Interpolates an entry into the array of orientation histograms that form
  the feature descriptor.

  @param hist 2D array of orientation histograms
  @param rbin sub-bin row coordinate of entry
  @param cbin sub-bin column coordinate of entry
  @param obin sub-bin orientation coordinate of entry
  @param mag size of entry
  @param d width of 2D array of orientation histograms
  @param n number of bins per orientation histogram
*/
static void interp_hist_entry( double*** hist, double rbin, double cbin, double obin, double mag, int d, int n )
{
  double d_r, d_c, d_o, v_r, v_c, v_o;
  double** row, * h;
  int r0, c0, o0, rb, cb, ob, r, c, o;

  r0 = cvFloor( rbin );
  c0 = cvFloor( cbin );
  o0 = cvFloor( obin );
  d_r = rbin - r0;
  d_c = cbin - c0;
  d_o = obin - o0;

  /*
    The entry is distributed into up to 8 bins.  Each entry into a bin
    is multiplied by a weight of 1 - d for each dimension, where d is the
    distance from the center value of the bin measured in bin units.
  */
  for( r = 0; r <= 1; r++ )
    {
      rb = r0 + r;
      if( rb >= 0  &&  rb < d )
	 {
	  v_r = mag * ( ( r == 0 )? 1.0 - d_r : d_r );
	  row = hist[rb];
	  for( c = 0; c <= 1; c++ )
	   {
	      cb = c0 + c;
	      if( cb >= 0  &&  cb < d )
		{
			  v_c = v_r * ( ( c == 0 )? 1.0 - d_c : d_c );
			  h = row[cb];
			  for( o = 0; o <= 1; o++ )
			  {
				  ob = ( o0 + o ) % n;
				  v_o = v_c * ( ( o == 0 )? 1.0 - d_o : d_o );
				  h[ob] += v_o;
			  } 
		}
	   }
	 }
    }
}



/*
  把直方图数组hist转换成最终的描述子向量。要经过3个步骤，直方图合并成向量，向量的单位化，大的bin值的限制，再次向量化。
  @param hist 2D array of orientation histograms
  @param d width of hist
  @param n bins per histogram
  @param feat feature into which to store descriptor
*/
static void hist_to_descr( double*** hist, int d, int n, struct feature* feat )
{
  int int_val, i, r, c, o, k = 0;

  for( r = 0; r < d; r++ )
    for( c = 0; c < d; c++ )
      for( o = 0; o < n; o++ )
	feat->descr[k++] = hist[r][c][o];

  feat->d = k; //128维向量
  normalize_descr( feat ); //描述子feat的归一化，处理之后的feat的二范数长度为1
  for (i = 0; i < k; i++)  //限制大的直方图的值，超过0.2的就限制为0.2了
  {
	  if (feat->descr[i] > SIFT_DESCR_MAG_THR)
		  feat->descr[i] = SIFT_DESCR_MAG_THR;
  }
  normalize_descr(feat); //再次归一化

  /* convert floating-point descriptor to integer valued descriptor */
  for( i = 0; i < k; i++ )
    {
      int_val = SIFT_INT_DESCR_FCTR * feat->descr[i];
      feat->descr[i] = MIN( 255, int_val );
    }
}


/*
  把向量feat进行归一化，使其二范数长度为1
  @param feat feature
*/
static void normalize_descr( struct feature* feat )
{
  double cur, len_inv, len_sq = 0.0;
  int i, d = feat->d;

  for( i = 0; i < d; i++ )
  {
      cur = feat->descr[i];
      len_sq += cur*cur;
  }
  len_inv = 1.0 / sqrt( len_sq );
  for( i = 0; i < d; i++ )
    feat->descr[i] *= len_inv;
}



/*
  Compares features for a decreasing-scale ordering.  Intended for use with
  CvSeqSort

  @param feat1 first feature
  @param feat2 second feature
  @param param unused

  @return Returns 1 if feat1's scale is greater than feat2's, -1 if vice versa,
    and 0 if their scales are equal
*/
static int feature_cmp( void* feat1, void* feat2, void* param )
{
  struct feature* f1 = (struct feature*) feat1;
  struct feature* f2 = (struct feature*) feat2;

  if( f1->scl < f2->scl )
    return 1;
  if( f1->scl > f2->scl )
    return -1;
  return 0;
}



/*
  De-allocates memory held by a descriptor histogram

  @param hist pointer to a 2D array of orientation histograms
  @param d width of hist
*/
static void release_descr_hist( double**** hist, int d )
{
  int i, j;

  for( i = 0; i < d; i++)
    {
      for( j = 0; j < d; j++ )
	free( (*hist)[i][j] );
      free( (*hist)[i] );
    }
  free( *hist );
  *hist = NULL;
}


/*
  De-allocates memory held by a scale space pyramid

  @param pyr scale space pyramid
  @param octvs number of octaves of scale space
  @param n number of images per octave
*/
static void release_pyr( IplImage**** pyr, int octvs, int n )
{
  int i, j;
  for( i = 0; i < octvs; i++ )
    {
      for( j = 0; j < n; j++ )
	cvReleaseImage( &(*pyr)[i][j] );
      free( (*pyr)[i] );
    }
  free( *pyr );
  *pyr = NULL;
}
