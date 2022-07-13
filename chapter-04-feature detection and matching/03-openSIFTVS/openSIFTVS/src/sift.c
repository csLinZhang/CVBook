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
   ������ͼ��img��ȡ�߶Ȳ��������㲢����SIFT����
   �ú����ķ���ֵΪimg�ϳ߶Ȳ���ĵ�ĸ���

   img������ͼ��
   feat��ָ�����飬�洢SIFT����������
   intvls����˹�߶ȿռ��У�ÿ��Ĳ�������Ӧ�ڽ̲��е�s��ÿ�����ղ���Ϊintvls+3
   sigma��ÿ���0��ĸ�˹�߶�
   @param cont_thr a threshold on the value of the scale space function
     \f$\left|D(\hat{x})\right|\f$, where \f$\hat{x}\f$ is a vector specifying
     feature location and scale, used to reject unstable features;  assumes
     pixel values in the range [0, 1]
   @param curv_thr threshold on a feature's ratio of principle curvatures
     used to reject features that are too edge-like
   descr_width�������ڼ���SIFT�����ӵľֲ����򻮷ֳ�descr_width*descr_width�Ŀ飬���ֵΪ4
   descr_hist_bins��ÿ��С�����ݶȷ���ֱ��ͼС�֣�bin������Ŀ�����ֵΪ8

   �������ͼ��img�гɹ���⵽��������ĸ���
*/
extern int sift_features( IplImage* img, struct feature** feat, int intvls, double sigma, double contr_thr, int curv_thr,
	int descr_width, int descr_hist_bins)
{
	//��˹�߶ȿռ��еĳ�ʼͼ����ֱ���Ϊimg��2������Ӧ��˹�߶�Ϊ1.6
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

	//��ʼ����˹�߶ȿռ�ĳ�ʼ�㣬sigmaΪ��ʼ��ĳ߶�Ϊ1.6
	init_img = create_init_img(img, sigma);
	//ȷ����˹�߶ȿռ����������Ϊÿ������һ��ֱ��ʵ�һ�룬������������޵�����ͼ��Ķ̱߳����й�
	octvs = log(MIN(init_img->width, init_img->height)) / log(2) - 2;
	//������˹�߶ȿռ��������octvs������intvlsΪÿ�������sigmaΪ��ʼ���˹�߶�
	gauss_pyr = build_gauss_pyr(init_img, octvs, intvls, sigma);
	//����DoG�߶ȿռ�
	dog_pyr = build_dog_pyr(gauss_pyr, octvs, intvls);

	storage = cvCreateMemStorage(0);
	
	//��DoG�߶ȿռ��������Ѱ�ҳ߶Ȳ���������
	features = scale_space_extrema(dog_pyr, octvs, intvls, contr_thr, curv_thr, storage);
	//����������ĳ߶ȣ���������֮ǰ�Ѿ�֪���������ڵĲ��Լ����ƫ���������Բ�ֵ�������ȷ�ĳ߶�ֵ
	calc_feature_scales(features, sigma, intvls);
	
	//��Ϊ�ڹ����߶ȿռ�ʱ����ʼͼ��ķֱ���������������ͼ���2�������������յ�������λ����������ͼ��Ϊ��׼�ģ������Ҫ����һ���������λ�úͳ߶�
	adjust_for_img_dbl(features);

	//����һ��Ϊֹ��features�д洢�����������������ͼ��ֱ��ʵ�λ�ã��������������߶ȡ�
	//Ҳ�洢�˸���������DoG�߶ȿռ��е�λ�ã�[octIndex, layIndex, r, c]
	//��һ��Ҫ��������������������ĳ���������ж���һ����������Ļ�����Ҫ���Ƽ��ݣ�ÿ����������������������к�������
	//Ҳ����˵����ĳ��λ�á�ĳ�������߶��ϣ��п��ܴ��ڼ������в�ͬ�������������
	calc_feature_oris(features, gauss_pyr);

	//Ŀǰ��features���鵱�е�ÿ��Ԫ�ش洢���������λ�ã������߶ȣ������߶�������Ͳ㣬������������Ϣ
	//���ڻ�����Щ��ϢҪΪÿ�������㹹��������
	//descr_widthֵΪ4��descr_hist_binsֵΪ8������SIFT�����ӵ�ά�Ⱦ���4*4*8=128
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
  �ú�������׼���ù����˹�߶ȿռ�ĵ�һ��ͼ�񣬸�ͼ���Ӧ�ĳ߶�Ϊ1.6.
  ͼ����ԭʼ����ͼ��2���ϲ����õ��������������ʹ������Ӧ�ĸ�˹�߶�Ϊ1.6
  img��ԭʼ����ͼ������Ҫ�Ը�ͼ����߶Ȳ���������
  sigma����˹�߶ȿռ��һ��ͼ������һ����ĳ߶ȣ�����Lowe���飬��ֵΪ1.6
*/
static IplImage* create_init_img( IplImage* img, double sigma )
{
  IplImage* gray, * dbl; //gray��ԭʼͼ��ת��֮��ĻҶ�ͼ��dbl��2���ϲ���ͼ��
  double sig_diff;

  gray = convert_to_gray32( img ); //��˹�߶ȿռ�ֻ�ܶԻҶ�ͼ������������ת�ɻҶ�ͼ��

  //����Lowe���飬��ԭʼ����ͼ�����2���ϲ����õ�dbl������������ĸ�˹�߶ȿռ�ĵ�һ���ͼ��dblʵ����������ͼ��ͨ�����Բ�ֵ�ϲ���2���Ľ��
  //����Lowe���飬��dbl���г߶�Ϊsigma = 1.6���˲���Lowe����ԭʼ����ͼ��img���������ͷ����ʱʵ�����Ѿ������˸�˹�˲�������߶�Ϊ0.5����
  //SIFT_INIT_SIGMA=0.5,�ǷŴ�2��ʱdbl�ġ��Դ����߶���Ӧ�ľͳ���2*SIFT_INIT_SIGMA������Ϊ��ʹ�˲����dbl�ĳ߶�Ϊ1.6,���Ƕ���ʩ�ӵĸ�˹�˲��ĳ߶�
  //ֻ��ҪΪsig_diff = sqrt(1.6^2-(SIFT_INIT_SIGMA*2)^2)
  sig_diff = sqrt( sigma * sigma - SIFT_INIT_SIGMA * SIFT_INIT_SIGMA * 4 );
  dbl = cvCreateImage( cvSize( img->width*2, img->height*2 ), IPL_DEPTH_32F, 1 ); 
  cvResize( gray, dbl, CV_INTER_CUBIC ); //2���ϲ�����������˫���β�ֵ
  cvSmooth( dbl, dbl, CV_GAUSSIAN, 0, 0, sig_diff, sig_diff ); //��dbl���и�˹�˲����˲�֮���dbl��Ӧ�ĸ�˹�߶�Ϊsigma=1.6
  cvReleaseImage( &gray );

  return dbl;
}

/*
 ������ͼ��ת����32bit�Ҷ�ͼ������ת���ɻҶ�ͼ�Ұ�ȡֵ��Χ��һ����[0,1]֮�ڣ�pixelValue/255.0

 img������ͼ�񣬿�����8-bit�Ҷ�ͼ��Ҳ������BGR��ɫͼ
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
  ������˹�߶ȿռ������

  base����˹�߶ȿռ�ĳ�ʼ�㣬��Ӧ�ڳ߶�1.6
  octvs������
  intvls��ÿ��ָ�����ÿ�����Ϊintvls+3
  sigma, ÿ���0��ĸ�˹����ˣ�Ҫע��ÿ���Ӧ��ĸ�˹�����ʵ������һ����ģ��������Ӧ��2���ĳ߶ȹ�ϵͨ���²���������

  ����gauss_pyr����˹�߶ȿռ�������������ܲ���Ϊoctvs x (intvls + 3)
*/
static IplImage*** build_gauss_pyr( IplImage* base, int octvs, int intvls, double sigma )
{
  IplImage*** gauss_pyr; //��˹������
  double *sig = (double*)calloc(intvls+3, sizeof(double));

  gauss_pyr = calloc( octvs, sizeof( IplImage** ) );
  for( int i = 0; i < octvs; i++ )
    gauss_pyr[i] = calloc( intvls + 3, sizeof( IplImage *) );

  /*
    Ԥ�ȼ����ÿ��ʹ�õĸ�˹����˵ı�׼��
	��Ҫע�⣺ÿ��ĸ�˹��׼���һ���ģ�������Ϊ���������Ӧ��ĳ߶�Ϊ2����ϵ������������Ŀռ�ֱ������ü���
	���Ը�˹�߶��ϷŴ������Ĺ�ϵ��ת��Ϊͼ��ռ�ֱ��ʼ������ˣ����ÿ���Ӧ��ʵ�ʾ���õĸ�˹��׼����һ���ġ�����Ҫ��������
	���������Ӧ��֮��ĸ�˹�߶ȹ�ϵΪ2����ϵ

	���е�i��õ���k+1��ʱ��ֻ��Ҫ�Ե�i����б�׼��Ϊsqrt(sigma_{i+1}^2 - sigma_{i}^2)�ĸ�˹�˲�
    sig[i+1]�洢���Ǵӵ�i������i+1��ʱ����Ҫ�Ե�i����еĸ�˹����ı�׼�i>0����
	��Ȼ��ÿ���0���ǲ��ý��о���ģ���һ��ĵ�0��������׼���õģ��߶�Ϊ1.6;
	�Ժ�ÿһ��ĵ�0������һ�鵹��������������1/2�²����õ������˹�߶�����һ�鵹��������һ��
  */
  double k = pow( 2.0, 1.0 / intvls ); //��˹�߶ȿռ��У���������֮��߶ȵı���Ϊk
  sig[0] = sigma; //ÿ���0��߶�
  sig[1] = sigma * sqrt( k*k- 1 ); //ÿ���1��߶�����
  for (int i = 2; i < intvls + 3; i++)
      sig[i] = sig[i-1] * k; //�ӵ�2�㿪ʼ��ÿ��߶�������������һ��������k

  for (int o = 0; o < octvs; o++) //������
  {
	  for (int i = 0; i < intvls + 3; i++) //���ڱ����㣬ÿ����intvls + 3��
	  {
		  if (o == 0 && i == 0) //��˹�߶ȿռ��ʼ��
			  gauss_pyr[o][i] = cvCloneImage(base);
		  else if (i == 0)/* ��ʼ�µ�һ���ˣ���0��Ϊ��һ�鵹�����������1/2�²��� */
			  gauss_pyr[o][i] = downsample(gauss_pyr[o - 1][intvls]);
		  else/*�ڱ����ڣ���i���ɶԵ�i-1��ʩ�ӱ�׼��Ϊsig[i]�ĸ�˹����õ�*/
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
  ��ͼ�����1/2������
*/
static IplImage* downsample( IplImage* img )
{
  IplImage* smaller = cvCreateImage( cvSize(img->width / 2, img->height / 2), img->depth, img->nChannels );
  cvResize( img, smaller, CV_INTER_LINEAR);

  return smaller;
}

/*
  //�Ӹ�˹�߶ȿռ������������DoG�߶ȿռ������
  gauss_pyr����˹�߶ȿռ������
  octvs����˹�߶ȿռ������������
  intvls����˹�߶ȿռ������ÿ��������ÿ�����Ϊintvls+3

  ����dog_pyr��ΪDOG�߶ȿռ䣬�����Ϊ octvs * (intvls + 2) 
*/
static IplImage*** build_dog_pyr( IplImage*** gauss_pyr, int octvs, int intvls )
{
  IplImage*** dog_pyr; //DoG�߶ȿռ��������ÿ����intvls+2��

  dog_pyr = calloc( octvs, sizeof( IplImage** ) );
  for(int i = 0; i < octvs; i++ )
    dog_pyr[i] = calloc( intvls + 2, sizeof(IplImage*) );

  for (int o = 0; o < octvs; o++)
  {
	  for (int i = 0; i < intvls + 2; i++)
	  {
		  dog_pyr[o][i] = cvCreateImage(cvGetSize(gauss_pyr[o][i]), IPL_DEPTH_32F, 1);
		  //DoG�ռ�����Ĳ��ɸ�˹�߶ȿռ��ж�Ӧ��������������õ�
		  cvSub(gauss_pyr[o][i + 1], gauss_pyr[o][i], dog_pyr[o][i], NULL);
	  }
  }
  return dog_pyr;
}

/*
  ��DoG�߶ȿռ��������Ѱ�ҳ߶Ȳ���������
  dog_pyr��DoG�߶ȿռ������
  octvs��DOG�߶ȿռ��е�����
  intvls��ÿ��������ÿ�����Ϊintvls+2
  contr_thr,���DoG�е����ӦֵС�ڸ���ֵ����õ㲻��������
  @param curv_thr high threshold on feature ratio of principal curvatures
  @param storage memory storage in which to store detected features

  ���س߶Ȳ��������㼯��
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
	for (int layerIndex = 1; layerIndex <= intvls; layerIndex++) //��layerIndex���ڵĲ����Ǳ���������⣬�ò�Ҫ�����������ϵĲ���бȽ�
	{
		for (int r = SIFT_IMG_BORDER; r < dog_pyr[octaveIndex][0]->height - SIFT_IMG_BORDER; r++) //rΪ������
		{
			for (int c = SIFT_IMG_BORDER; c < dog_pyr[octaveIndex][0]->width - SIFT_IMG_BORDER; c++)//cΪ������
			{
				/* �õ����Ӧֵ�ľ���ֵ��Ҫ����һ����ֵprelim_contr_thr*/
				if (ABS(pixval32f(dog_pyr[octaveIndex][layerIndex], r, c)) > prelim_contr_thr)
				{
					if (is_extremum(dog_pyr, octaveIndex, layerIndex, r, c)) //���[o,layerIndex,r,c]��Ϊһ���ֲ���ֵ��
					{
						//��[octaveIndex, layerIndex, r, c]��ȷ��������λ���ϵļ�ֵ�����λ�þ���
						feat = interp_extremum(dog_pyr, octaveIndex, layerIndex, r, c, intvls, contr_thr);
						if (feat) //���������ɹ�
						{
							ddata = feat_detection_data(feat);
							//�жϸ��������Ƿ���ίԱͼ��edge֮�ϣ�����ǵĻ����õ㲻����Ϊ��������
							//ע�⣬�ǲ��Ǳ�Ե����ж����ڳ�ʼ��ɢλ�����ж��ģ���������λ�þ���֮��ĵ��Ͻ��еģ���Ϊλ�þ���֮��ĵ�
							//�ڳ߶ȿռ��е����궼���������ˣ���������Hessian����ͷǳ�������
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
  �ж���DoG�߶ȿռ�������У�����octv���ڲ�intvl����[r,c]��ֵ�Ƿ�Ϊ��ֵ��,�õ���Ҫ�����ڳ߶ȿռ��е�26���ھӽ��бȽ�

  dog_pyr��DoG�߶ȿռ������
  octv�����жϵĵ����ڵ��������
  layer�����жϵĵ�������Ĳ������
  r�����жϵĵ���к�
  c�����жϵĵ���к�

  ����1����˵���õ�Ϊ��ֵ�㣬����0˵�����Ǽ�ֵ��
*/
static int is_extremum( IplImage*** dog_pyr, int octvIndex, int layerIndex, int r, int c )
{
  double val = pixval32f( dog_pyr[octvIndex][layerIndex], r, c ); //���ظõ���DoG�߶ȿռ��е�ֵ

  /*DoG�˲���Ϊ���ֵ�˲��������õ���Ӧֵ����0��������Ҫ�ж����Ƿ�Ϊ����ֵ */
  if( val > 0 )
    {
	  //������26���߶ȿռ��е��ھӽ��бȶԣ�ֻҪ�б�����ģ���˵�����϶����Ǽ�ֵ��
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
  else/*�ж����Ƿ�Ϊ�ֲ���Сֵ */
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
  ��DoG�г�ʼ�õ��ľֲ���ֵ���λ�ý��о������Եõ�����׼ȷ�ļ�ֵ��λ�ã��ռ�λ���Լ��߶�λ�ã�
  ���ڵ�ԭ����ǶԸõ�ֲ��߶ȿռ���Ϣ���ж���̩��չ����Ȼ����ֲ���Сֵ�㣬�˹�������������5��

  @param dog_pyr��DoG�߶ȿռ������
  @param octvIndex����ǰ��ʼ��ֵ����DoG�߶ȿռ�������е�������
  @param layerIndex����ǰ�����ʼ��ֵ������octv�еĲ�����
  @param r feature's image row
  @param c feature's image column
  @param intvls��DoG�߶ȿռ��������ÿ���ڵļ����
  @param contr_thr,����õ�ĶԱȶȵ�����ֵcontr_thr����Ҳ����Ϊ�õ�Ϊ������

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
  
  while(iterationTimes < SIFT_MAX_INTERP_STEPS ) //��ೢ��5��
    {
	  //�Ե�ǰ������ֵ��λ��[octv, layerIndex, r, c]���ж��β�ֵ���õ������� ��xi, xr, xc��
      interp_step( dog_pyr, octvIndex, layerIndex, r, c, &xi, &xr, &xc );
      if( ABS( xi ) < 0.5  &&  ABS( xr ) < 0.5  &&  ABS( xc ) < 0.5 ) //������ų߶ȿռ�����ά�ȵĸ�������С��0.5����˵�����ø�����
		break;
      
	  //���µ��µ�DOG�߶ȿռ�������λ�õ�
      c += cvRound( xc );
      r += cvRound( xr );
	  layerIndex += cvRound( xi ); //ע�⣬Lowe��������˵���Գ߶�sigma�󵼣����ǲ�׼ȷ�ģ�ʵ�����ǶԳ߶�����Ӧ�Ĳ��󵼣����ڲ��������ļ��Ϊ1������
	 
      //�Ը���֮����µ�����λ�õ����һЩ��飬���������������Ļ������糬���˱�������ı߽磩������Ϊ����ʧ��
      if(layerIndex < 1  || layerIndex > intvls  || c < SIFT_IMG_BORDER  || r < SIFT_IMG_BORDER  
		  || c >= dog_pyr[octvIndex][0]->width - SIFT_IMG_BORDER  || r >= dog_pyr[octvIndex][0]->height - SIFT_IMG_BORDER )
	  {
		  return NULL;
	  }
	  iterationTimes++;
    }
  
  /* ensure convergence of interpolation */
  //����жϺܹؼ�������ζ�������whileѭ���е�breakһ��Ҫ���������������������Ϊû������
  //break�����ˣ�˵�����һ���������xr��xc��û�и��µ�r,c,��layerIndex֮��
  if(iterationTimes >= SIFT_MAX_INTERP_STEPS )
    return NULL;
  
  //�����ڲ�ֵ�㴦��DoG��Ӧֵ�������ֵ��С����õ㲻����Ϊ��������
  contr = interp_contr( dog_pyr, octvIndex, layerIndex, r, c, xi, xr, xc );
  if( ABS( contr ) < contr_thr / intvls )
    return NULL;

  feat = new_feature();
  ddata = feat_detection_data( feat );
  feat->img_pt.x = feat->x = ( c + xc ) * pow( 2.0, octvIndex); //���²�����Ŀռ�ֱ��ʷ��ص�ԭʼͼ��2����С�Ŀռ�λ��
  feat->img_pt.y = feat->y = ( r + xr ) * pow( 2.0, octvIndex);
  //ddata��������feat�ڳ߶ȿռ��е���Ϣ������һ����С��У�����һ�顢��һ�㡢�ǲ�ƫ����
  //ע�⣺ddata�е�r,c,layerIndex�����һ�ξ�������֮ǰ����DoG�߶ȿռ��е�λ����Ϣ����Ϊ������Ҳ����˵
  //���һ��ƫ���������ǻ��ڸõ�[r,c,layerIndex]����̩��չ���ġ�֮����Ҫ��¼�������λ�ã���Ϊ�˺����жϱ�Ե��͹��������ӷ���
  ddata->r = r; 
  ddata->c = c;
  ddata->octv = octvIndex;
  ddata->intvl = layerIndex;
  ddata->subintvl = xi; //��¼�����ǲ��ƫ����

  return feat;
}

/*
  Performs one step of extremum interpolation.  Based on Eqn. (3) in Lowe's
  paper.
  �ӵ�ǰ������ֵ��[octv, intvl, r, c]���㼫ֵ��λ�ö�Ӧ�ĸ�����
  @param dog_pyr��DoG�߶ȿռ������
  @param octvIndex�������� octave of scale space
  @param layerIndex�����ڲ�����
  @param r�������� 
  @param c��������
  @param xi���Բ�ĸ�����
  @param xr�����еĸ�����
  @param xc�����еĸ�����
*/

static void interp_step( IplImage*** dog_pyr, int octvIndex, int layerIndex, int r, int c, double* xi, double* xr, double* xc )
{
  CvMat* dD, * H, * H_inv, X;
  double x[3] = { 0 };
  
  //����һ����ά������DoG(x, y, sigma)�ڵ�[octvIndex, layerIndex, r, c]��{ dDoG / dx, dDoG / dy, dDog / dl }^T
  dD = deriv_3D( dog_pyr, octvIndex, layerIndex, r, c );

  //����DoG�߶ȿռ���[octvIndex, layerIndex, r, c]���ĺ�ɭ����
  H = hessian_3D( dog_pyr, octvIndex, layerIndex, r, c );
  H_inv = cvCreateMat( 3, 3, CV_64FC1 ); //H_invΪ��ɭ����������
  cvInvert( H, H_inv, CV_SVD );
  cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
  cvGEMM( H_inv, dD, -1, NULL, 0, &X, 0 ); //- H^(-1)*Gradient,��������������X��
  
  cvReleaseMat( &dD );
  cvReleaseMat( &H );
  cvReleaseMat( &H_inv );

  *xi = x[2];
  *xr = x[1];
  *xc = x[0];
}

/*
  ��DoG�߶ȿռ��У�����DoG(x,y,l)�ڵ㣨x,y,l������ƫ����,lΪ������
  @param dog_pyr��DoG�߶ȿռ������
  @param octvIndex�� ���������
  @param layerIndex��������ڲ�����
  @param r�����������
  @param c�����������

  ����һ����ά������DoG(x,y,sigma)�ڵ�[octvIndex, layerIndex, r, c]��{ dDoG/dx, dDoG/dy, dDog/l }^T 
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
  ����DoG(x,y,sigma)�ĺ�ɭ����
  dog_pyr��DoG�߶ȿռ�
  @param octvIndex,������
  @param layerIndex��������
  @param r��������
  @param c��������

  @���ص�[octvIndex, layerIndex, r, c]��DoG(x,y,layerIndex)�ĺ�ɭ����

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
  ���㼫ֵ�㴦��DoG��Ӧֵ��ע�������ֵ��λ���Ǿ�����������֮���
  @param dog_pyr��DoG�߶ȿռ������
  @param octv,������
  @param layerIndex��ԭʼ������ֵ�㴦�Ĳ�����
  @param r��ԭʼ������ֵ�㴦��������
  @param c��ԭʼ������ֵ�㴦��������
  @param xi����ƫ����
  @param xr����ƫ����
  @param xc����ƫ����

  ���ز�ֵ�㴦��DoG��Ӧֵ
*/
static double interp_contr( IplImage*** dog_pyr, int octvIndex, int layerIndex, int r, int c, double xi, double xr, double xc )
{
  CvMat* dD, X, T;
  double t[1], x[3] = { xc, xr, xi };

  //ʵ�ֽ̲Ĺ�ʽ4-16
  cvInitMatHeader( &X, 3, 1, CV_64FC1, x, CV_AUTOSTEP );
  cvInitMatHeader( &T, 1, 1, CV_64FC1, t, CV_AUTOSTEP );
  dD = deriv_3D( dog_pyr, octvIndex, layerIndex, r, c );
  cvGEMM( dD, &X, 1, NULL, 0, &T,  CV_GEMM_A_T );
  cvReleaseMat( &dD );

  return pixval32f( dog_pyr[octvIndex][layerIndex], r, c ) + t[0] * 0.5;
}

/*
  ���첢��ʼ��һ��SIFT�������������������λ����Ϣ������������
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
  �ж�һ���������������Ƿ�λ��ͼ���Ե֮�ϡ�ע�⣺���жϵĵ�ʵ�������뾫�������������λ�õ�DoG�߶ȿռ�������λ�ô��ĵ�

  @param dog_img��DoG�߶ȿռ��е�һ�㣬�ڸò��[r,c]λ��Ϊ���жϵ�������
  @param r����������DoG�ò��е���
  @param c����������DoG�ò��е���
  @param curv_thr����ֵ����ֵ����David Lowe��������Ϊ10
*/
static int is_too_edge_like( IplImage* dog_img, int r, int c, int curv_thr )
{
  double d, dxx, dyy, dxy, tr, det;

  /* ����õ��Hessian�����е�Ԫ�� */
  d = pixval32f(dog_img, r, c);
  dxx = pixval32f( dog_img, r, c+1 ) + pixval32f( dog_img, r, c-1 ) - 2 * d;
  dyy = pixval32f( dog_img, r+1, c ) + pixval32f( dog_img, r-1, c ) - 2 * d;
  dxy = ( pixval32f(dog_img, r+1, c+1) - pixval32f(dog_img, r+1, c-1) - pixval32f(dog_img, r-1, c+1) + pixval32f(dog_img, r-1, c-1) ) / 4.0;
  tr = dxx + dyy; //Hessian����ļ�
  det = dxx * dyy - dxy * dxy; //Hessian���������ʽ

  /*�������ʽС��0��˵���õ��������������ţ���õ㲻�Ǿֲ���ֵ�㣬���� */
  if( det <= 0 )
    return 1;

  if( tr * tr / det < ( curv_thr + 1.0 )*( curv_thr + 1.0 ) / curv_thr )
    return 0;
  return 1;
}

/*
  ����������ĳ߶ȣ���������֮ǰ�Ѿ�֪���������ڵĲ��Լ����ƫ���������Բ�ֵ�������ȷ�ĳ߶�ֵ
  @param features array of features
  @param sigma,��˹�߶ȿռ�ĳ�ʼ�߶ȣ�Ҳ��ÿһ��ĳ�ʼ�߶ȣ�ע�⣺ÿһ��ĳ�ʼ�߶�sigma��һ���ģ�2����ϵ��ͨ���ֱ��ʼ��������ֵ�
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
      intvl = ddata->intvl + ddata->subintvl; //��ȷ���߶ȿռ��ǲ�
	  //�����������������߶�
      feat->scl = sigma * pow( 2.0, ddata->octv + intvl / intvls );
	  //ddata->scl_octv���Ǹ��������������һ���ʼ��ĳ߶ȣ�����Ҳ�Ǿ�ȷ���ǲ�ĳ߶�
      ddata->scl_octv = sigma * pow( 2.0, intvl / intvls );
    }
}

/*
  ��Ϊ�ڹ����߶ȿռ�ʱ����ʼͼ��ķֱ���������������ͼ���2�������������յ�������λ����������ͼ��Ϊ��׼��
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

/*��һ��Ҫ��������������������ĳ���������ж���һ����������Ļ�����Ҫ���Ƽ��ݣ�ÿ����������������������к�������
Ҳ����˵����ĳ��λ�á�ĳ�������߶��ϣ��п��ܴ��ڼ������в�ͬ�������������
  @features��һ�����飬ÿһ��Ԫ����һ������
  @gauss_pyr����˹�߶ȿռ�
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
      cvSeqPopFront( features, feat ); //��������ǰ�features���鵱���������һ��ɾ������������ʱ�洢��feat��
      ddata = feat_detection_data( feat );
	  //��������������ж�Ҫ�����ݶȷ���ֱ��ͼ���жϣ�����ݶȷ���ֱ��ͼҪ���������߶ȶ�Ӧ�ĸ�˹���Ͻ���
      hist = ori_hist(gauss_pyr[ddata->octv][ddata->intvl], ddata->r, ddata->c, SIFT_ORI_HIST_BINS, cvRound( SIFT_ORI_RADIUS * ddata->scl_octv ), SIFT_ORI_SIG_FCTR * ddata->scl_octv );
	  for (int j = 0; j < SIFT_ORI_SMOOTH_PASSES; j++)
	  {
		  //��ֱ��ͼ����2��ƽ��
		  smooth_ori_hist(hist, SIFT_ORI_HIST_BINS);
	  }
      omax = dominant_ori(hist, SIFT_ORI_HIST_BINS ); //�ҵ��ݶȷ���ֱ��ͼhist������ֵ
      add_good_ori_features(features, hist, SIFT_ORI_HIST_BINS, omax * SIFT_ORI_PEAK_RATIO, feat);
      free( ddata );
      free( feat );
      free( hist );
  }
}

/*
  ��ͼ��img��һ������ݶȷ���ֱ��ͼ
  img������ͼ��
  r������������
  c������������
  n������ֱ��ͼ������Ŀ
  rad������ֱ��ͼʱʹ�õ�����뾶, 4.5*characterisitc scale
  sigma, �ڼ����ݶ�ֱ��ͼʱ��ÿ��Ĺ�����Ҫ���ոõ㵽���ľ���ĸ�˹��Ȩ���ø�˹������stdΪsigma= 1.5*characterisitc scale

  ����һ�����飬��ʾ�ݶȷ���ֱ��ͼ���Ƕȷ�ΧΪ0~2PI
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
		  if (calc_grad_mag_ori(img, r + i, c + j, &mag, &ori)) //����õ��ݶȷ�ֵ�ͽǶ�
		  {
			  w = exp(-(i*i + j * j) / exp_denom); //�õ�(i,j)���ݶ�ֱ��ͼ�Ĺ���Ҫ������������ĵ�ľ����˹��Ȩ
			  //Ҫע�⣬ori��ȡֵ��ΧΪ[-PI, PI], ������Ҫ����CV_PI�����䷶Χ������[0, 2PI]
			  //�ȰѽǶȷ�Χת����(0,1)֮�䣬Ȼ�����n�������������ĸ�bin����
			  bin = cvRound(n * (ori + CV_PI) / PI2); 
			  bin = (bin < n) ? bin : 0;
			  hist[bin] += w * mag; //��Ȩ���¶�Ӧ��binֵ
		  }
	  }
  }
  return hist;
}

/*
  ����ͼ����һ����ݶȷ�ֵ��Ƕȣ��Ƕȵķ�ΧΪ[-PI, PI]
  @param img image
  @param r pixel row
  @param c pixel col
  @param mag output as gradient magnitude at pixel (r,c)
  @param ori output as gradient orientation at pixel (r,c),��ȡֵ��ΧΪ[-PI, PI]

  @return Returns 1 if the specified pixel is a valid one and sets mag and
    ori accordingly; otherwise returns 0
*/
static int calc_grad_mag_ori( IplImage* img, int r, int c, double* mag, double* ori )
{
  double dx, dy;

  if( r > 0  &&  r < img->height - 1  &&  c > 0  &&  c < img->width - 1 ) //���б߽���ԣ�̫����ͼ��߽�ĵ�Ͳ�������
    {
      dx = pixval32f( img, r, c+1 ) - pixval32f( img, r, c-1 );
      dy = pixval32f( img, r-1, c ) - pixval32f( img, r+1, c );
      *mag = sqrt( dx*dx + dy*dy ); //�ݶȵķ�ֵ
      *ori = atan2( dy, dx ); //���ȷ�Χ��[-PI, PI]
      return 1;
    }
  else
    return 0;
}


/*
  ���ݶȷ���ֱ��ͼ���и�˹ƽ��
  hist���ݶȷ���ֱ��ͼ
  n��ֱ��ͼ��bin����Ŀ
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
  Ѱ���ݶȷ���ֱ��ͼ�е�����ֵ
  hist�� an orientation histogram
  n�� number of bins

  �����ݶȷ���ֱ��ͼhist�е�����ֵ
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
  ����õ���ݶȷ���ֱ��ͼ�ж�������򣨶�Ӧ��bin��ֵ�������ֵ��80%�������ÿһ�������򶼸���һ�����������
  Ҳ����˵���ڸõ㴦����ͬ���������߶��£��м�����ͬ�������㣬���Ǿ���������ͬ��������������ƥ����ȶ��ԡ�

  features����������
  hist����ǰ�õ���ݶȷ���ֱ��ͼ
  n��ֱ��ͼ��bin����Ŀ
  mag_thr��һ����ֵ�����ֱ��ͼĳ��bin��ֵ������ֵ������һ������
  feat����ǰ����������Ѿ�����һ��feature��������õ��ж��������Ļ�����Ҫ���ƶ��feature,�ʹ�feat���Ƶ�
*/
static void add_good_ori_features( CvSeq* features, double* hist, int n, double mag_thr, struct feature* feat )
{
  struct feature* new_feat;
  double bin, PI2 = CV_PI * 2.0;
  int l, r;

  for( int binIndex = 0; binIndex < n; binIndex++ )
    {
      l = (binIndex == 0 )? n - 1 : binIndex -1; //��ǰbinIndex����ߵ�binIndex������ǰbinIndexΪ0�������binIndexΪn-1
      r = (binIndex + 1 ) % n; //��ǰbinIndex�ұߵ�binIndex������ǰbinIndexΪn-1�����ұ�binIndexΪ0
      
	  //��ǰbinIndex����֮��Ϊhist�е�һ���ֲ���ֵ���Ҵ�����ֵmag_thr
	  //������Ҫ�������з����ֵ������feat����һ�ݣ����ɶ��������
      if( hist[binIndex] > hist[l]  &&  hist[binIndex] > hist[r]  &&  hist[binIndex] >= mag_thr )
	  {
		  bin = binIndex + interp_hist_peak(hist[l], hist[binIndex], hist[r] );
		  bin = ( bin < 0 )? n + bin : ( bin >= n )? bin - n : bin;
		  new_feat = clone_feature( feat );
		  new_feat->ori = ( ( PI2 * bin ) / n ) - CV_PI; //���յ�������Χ��[-PI, PI]֮��
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
  ����SIFT����������
  features���鵱�е�ÿ��Ԫ�ش洢���������λ�ã������߶ȣ������߶�������Ͳ㣬������������Ϣ
  descr_widthֵΪ4��descr_hist_binsֵΪ8������SIFT�����ӵ�ά�Ⱦ���4*4*8=128
  gauss_pyr����˹�߶ȿռ䣬SIFT�ļ�����Ҫ��������������ĸ�˹�߶Ȳ��������е�
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
  ����SIFT����������

  img����˹�߶ȿռ��еĶ�Ӧ�ڸ��������һ��
  r��c,�ڸø�˹�����������λ��
  ori���������������
  scl,���ڸø�˹�߶Ȳ������ڸ����0��ĳ߶�
  descr_width�� width of 2d array of orientation histograms
  descr_hist_bins�� bins per orientation histogram

  ����һ��  descr_width x descr_width���飬ÿ��Ԫ��Ϊ n-bin���ݶȷ���ֱ��ͼ
*/
static double*** descr_hist( IplImage* img, int r, int c, double ori, double scl, int descr_width, int descr_hist_bins)
{
  double*** hist;
  double cos_t, sin_t, hist_width, exp_denom, r_rot, c_rot, grad_mag, grad_ori, w, rbin, cbin, obin, bins_per_rad, PI2 = 2.0 * CV_PI;
  int radius;

  hist = calloc(descr_width, sizeof( double** ) );
  //����ֱ��ͼ����ռ䣬4*4��ֱ��ͼ��ÿ��ֱ��ͼ8��bin
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
  hist_width = SIFT_DESCR_SCL_FCTR * scl; //3sigmaԭ��ȷ�����ڼ��������ӵ�����뾶
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
		  rbin = r_rot + descr_width / 2 - 0.5; //ȷ����ǰ��һ��Ӧ������hist��һ��bin��hist��4*4�Ķ�ά����ָ�룬ÿ��Ԫ����һ��ָ��
		  //8ά������ָ��
		  cbin = c_rot + descr_width / 2 - 0.5;

		  if (rbin > -1.0  &&  rbin < descr_width  &&  cbin > -1.0  &&  cbin < descr_width)
		  {
			  if (calc_grad_mag_ori(img, r + i, c + j, &grad_mag, &grad_ori))
			  {
				  grad_ori -= ori; //�����һ��
				  while (grad_ori < 0.0) //����ֵҪͳһ��[0,2PI]
					  grad_ori += PI2;
				  while (grad_ori >= PI2)
					  grad_ori -= PI2;

				  obin = grad_ori * bins_per_rad; //�ҵ���ǰ�������ݶȷ���Ӧ�����ڷ���ֱ��ͼ�ĸ�bin����
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
  ��ֱ��ͼ����histת�������յ�������������Ҫ����3�����裬ֱ��ͼ�ϲ��������������ĵ�λ�������binֵ�����ƣ��ٴ���������
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

  feat->d = k; //128ά����
  normalize_descr( feat ); //������feat�Ĺ�һ��������֮���feat�Ķ���������Ϊ1
  for (i = 0; i < k; i++)  //���ƴ��ֱ��ͼ��ֵ������0.2�ľ�����Ϊ0.2��
  {
	  if (feat->descr[i] > SIFT_DESCR_MAG_THR)
		  feat->descr[i] = SIFT_DESCR_MAG_THR;
  }
  normalize_descr(feat); //�ٴι�һ��

  /* convert floating-point descriptor to integer valued descriptor */
  for( i = 0; i < k; i++ )
    {
      int_val = SIFT_INT_DESCR_FCTR * feat->descr[i];
      feat->descr[i] = MIN( 255, int_val );
    }
}


/*
  ������feat���й�һ����ʹ�����������Ϊ1
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
