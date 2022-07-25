// monoCalib.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/calib3d/calib3d_c.h>
#include "myLM.h"

using namespace std;
using namespace cv;

int cvFindHomography(const CvMat* _src, const CvMat* _dst, CvMat* __H, int method CV_DEFAULT(0),
	double ransacReprojThreshold CV_DEFAULT(3), CvMat* _mask CV_DEFAULT(0), int maxIters CV_DEFAULT(2000),
	double confidence CV_DEFAULT(0.995))
{
	cv::Mat src = cv::cvarrToMat(_src), dst = cv::cvarrToMat(_dst);

	if (src.channels() == 1 && (src.rows == 2 || src.rows == 3) && src.cols > 3)
		cv::transpose(src, src);
	if (dst.channels() == 1 && (dst.rows == 2 || dst.rows == 3) && dst.cols > 3)
		cv::transpose(dst, dst);

	if (maxIters < 0)
		maxIters = 0;
	if (maxIters > 2000)
		maxIters = 2000;

	if (confidence < 0)
		confidence = 0;
	if (confidence > 1)
		confidence = 1;

	const cv::Mat H = cv::cvarrToMat(__H), mask = cv::cvarrToMat(_mask);
	cv::Mat H0 = cv::findHomography(src, dst, method, ransacReprojThreshold,
		_mask ? cv::_OutputArray(mask) : cv::_OutputArray(), maxIters,
		confidence);

	if (H0.empty())
	{
		cv::Mat Hz = cv::cvarrToMat(__H);
		Hz.setTo(cv::Scalar::all(0));
		return 0;
	}
	H0.convertTo(H, H.type());
	return 1;
}

//objectPoints，所有标定板图像物空间标定板角点世界坐标
//imagePoints，所有标定板图像的图像空间角点图像坐标
//npoints，所有标定板图像的信息，对于每张图像存储了角点个数
//cameraMatrix最终存储了内参矩阵
void cvInitIntrinsicParams2D(const CvMat* objectPoints, const CvMat* imagePoints, const CvMat* npoints, CvSize imageSize, CvMat* cameraMatrix)
{
	Ptr<CvMat> matA, _b, _allH;

	int i, j, pos, nimages, ni = 0;
	double a[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 1 }; //内参矩阵，
	double H[9] = { 0 }, f[2] = { 0 }; //H是标定板物理平面到图像平面的homography,f是fx,fy
	CvMat _a = cvMat(3, 3, CV_64F, a);
	CvMat matH = cvMat(3, 3, CV_64F, H);
	CvMat _f = cvMat(2, 1, CV_64F, f);

	//标定板图像张数
	nimages = npoints->rows + npoints->cols - 1;

	//matA为线性方程组系数矩阵，每个图像产生2个方程，每个方程2个未知数，分别对应1/fx^2和
	//1/fy^2，所以其dimension为[2*nimages,2]
	matA.reset(cvCreateMat(2 * nimages, 2, CV_64F));
	//_b为线性方程组右边的常数列，每个图像产生2个方程，所有b为[2*nimages, 1]
	_b.reset(cvCreateMat(2 * nimages, 1, CV_64F));

	//这个内参初始化算法不会计算cx, cy，所以cx,cy直接估计为图像中点
	a[2] = (!imageSize.width) ? 0.5 : (imageSize.width - 1)*0.5;
	a[5] = (!imageSize.height) ? 0.5 : (imageSize.height - 1)*0.5;

	//存储所有标定板到图像的homography matrix，每个Homography有9个元素
	_allH.reset(cvCreateMat(nimages, 9, CV_64F)); 

	// extract vanishing points in order to obtain initial value for the focal length
	for (i = 0, pos = 0; i < nimages; i++, pos += ni)
	{
		//Ap存储由当前图像产生的线性方程组的系数矩阵，2个方程，2个未知数，所以是4个数
		double* Ap = matA->data.db + i * 4;
		double* bp = _b->data.db + i * 2;
		ni = npoints->data.i[i];

		//h,v,d1,d2分别是四个无穷远点的图像坐标；n[i]最终存储了4个齐次坐标的模长倒数，为了
		//归一化使用
		double h[3], v[3], d1[3], d2[3];
		double n[4] = { 0,0,0,0 };

		//matM存储了标定板物点坐标
		//_m存储了标定板这张图像所对应的特征点图像坐标
		CvMat _m, matM;
		cvGetCols(objectPoints, &matM, pos, pos + ni);
		cvGetCols(imagePoints, &_m, pos, pos + ni);

		//计算这张标定图像物点到图像点坐标的单应矩阵
		cvFindHomography(&matM, &_m, &matH);
		//matH是计算完成的单应矩阵，这个矩阵在构造的时候把数据关联到了H，
		//H是个9维数组，关联存储了matH的数据。所以这里把H中的数据拷贝到
		//_allH中
		memcpy(_allH->data.db + i * 9, H, sizeof(H));

		//这里是要把图像坐标系原点移到主点位置
		/*    H0  H1  H2
		  H = H3  H4  H5
		      H6  H7  H8
		*/
		H[0] -= H[6] * a[2]; 
		H[1] -= H[7] * a[2]; 
		H[3] -= H[6] * a[5]; 
		H[4] -= H[7] * a[5]; 

		//h,v,d1,d2分别是四个无穷远点的图像坐标
		//d1 = (h+v)/2; d2=(h-v)/2
		//这3次循环是填充它们的值，由于齐次坐标是3个分量，所以循环三次
		//每次填充一个分量
		//同时，与Matlab版本完全相同，把最终的4个齐次坐标都进行了长度归一化，比如h=h/||h||
		//所以这个循环中n[0]最终是h的各分量平方和；n[1]最终是v的各分量平方和;
		//n[2]最终是d1的各分量平方和；n[3]最终是d2的各分量平方和;
		for (j = 0; j < 3; j++)
		{
			double t0 = H[j * 3];
			double t1 = H[j * 3 + 1];
			h[j] = t0; 
			v[j] = t1;
			d1[j] = (t0 + t1)*0.5;
			d2[j] = (t0 - t1)*0.5;
			n[0] += t0 * t0; 
			n[1] += t1 * t1;
			n[2] += d1[j] * d1[j]; 
			n[3] += d2[j] * d2[j];
		}

		//计算h,v,d1,d2四个齐次坐标的模长的倒数，为归一化做准备
		//n[0]=1/|h|, n[1]=1/|v|, n[2]=1/|d1|, n[3]=1/|d2|
		for (j = 0; j < 4; j++)
			n[j] = 1. / std::sqrt(n[j]);

		//对h,v,d1,d2四个无穷远点的图像坐标进行坐标归一化
		for (j = 0; j < 3; j++)
		{
			h[j] *= n[0]; 
			v[j] *= n[1];
			d1[j] *= n[2]; 
			d2[j] *= n[3];
		}

		//Ap是线性方程组的系数;bp是右侧的列向量
		//Ap*X=bp; 每张图像都会加入2个新的方程
		Ap[0] = h[0] * v[0]; 
		Ap[1] = h[1] * v[1];
		Ap[2] = d1[0] * d2[0]; 
		Ap[3] = d1[1] * d2[1];
		bp[0] = -h[2] * v[2]; //-c1c2
		bp[1] = -d1[2] * d2[2];//-c3c4
	}

	cvSolve(matA, _b, &_f, CV_NORMAL + CV_SVD);
	a[0] = std::sqrt(fabs(1. / f[0]));
	a[4] = std::sqrt(fabs(1. / f[1]));
	
	cvConvert(&_a, cameraMatrix);
}

//此函数把图像坐标系下的点_src转换到归一化平面坐标系，结果为_dst
//_cameraMatrix为内参矩阵
static void cvTransform2NormalizedPlaneInternal(const CvMat* _src, CvMat* _dst, const CvMat* _cameraMatrix)
{
	//A为内参矩阵，k为distortion coefficient
	double A[3][3], k[14] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
	CvMat matA = cvMat(3, 3, CV_64F, A);
	//matA为内参矩阵
	cvConvert(_cameraMatrix, &matA);

	const CvPoint2D64f* srcd = (const CvPoint2D64f*)_src->data.ptr;
	CvPoint2D64f* dstd = (CvPoint2D64f*)_dst->data.ptr;

	//两个焦距
	double fx = A[0][0];
	double fy = A[1][1];
	double ifx = 1. / fx;
	double ify = 1. / fy;
	//主点位置
	double cx = A[0][2];
	double cy = A[1][2];

	//n为点的个数
	int n = _src->rows + _src->cols - 1;
	for (int i = 0; i < n; i++)
	{
		double x, y, x0 = 0, y0 = 0;
		x = srcd[i].x;
		y = srcd[i].y;

		//此时的x,y是u,v对应的归一化平面坐标
		x = (x - cx)*ifx;
		y = (y - cy)*ify;

		dstd[i].x = x;
		dstd[i].y = y;
	}
}

int cvRodrigues2(const CvMat* src, CvMat* dst, CvMat* jacobian CV_DEFAULT(0))
{
	int depth, elem_size;
	int i, k;
	double J[27] = { 0 };
	CvMat matJ = cvMat(3, 9, CV_64F, J);

	depth = CV_MAT_DEPTH(src->type);
	elem_size = CV_ELEM_SIZE(depth);

	if (src->cols == 1 || src->rows == 1)
	{
		int step = src->rows > 1 ? src->step / elem_size : 1;
		Point3d r;
		if (depth == CV_32F)
		{
			r.x = src->data.fl[0];
			r.y = src->data.fl[step];
			r.z = src->data.fl[step * 2];
		}
		else
		{
			r.x = src->data.db[0];
			r.y = src->data.db[step];
			r.z = src->data.db[step * 2];
		}

		double theta = norm(r);

			double c = cos(theta);
			double s = sin(theta);
			double c1 = 1. - c;
			double itheta = theta ? 1. / theta : 0.;

			r *= itheta;

			//此时的r为单位化向量，相当于公式里面的n
			//相当于公式里面的n*nt,t为转置
			Matx33d rrt(r.x*r.x, r.x*r.y, r.x*r.z, r.x*r.y, r.y*r.y, r.y*r.z, r.x*r.z, r.y*r.z, r.z*r.z);
			//相当于公式里面的n(hat)
			Matx33d r_x(0, -r.z, r.y,
				r.z, 0, -r.x,
				-r.y, r.x, 0);

			// R = cos(theta)*I + (1 - cos(theta))*r*rT + sin(theta)*[r_x]
			Matx33d R = c * Matx33d::eye() + c1 * rrt + s * r_x;

			Mat(R).convertTo(cvarrToMat(dst), dst->type);

			if (jacobian)
			{
				const double I[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
				double drrt[] = { r.x + r.x, r.y, r.z, r.y, 0, 0, r.z, 0, 0,
								  0, r.x, 0, r.x, r.y + r.y, r.z, 0, r.z, 0,
								  0, 0, r.x, 0, 0, r.y, r.x, r.y, r.z + r.z };
				double d_r_x_[] = { 0, 0, 0, 0, 0, -1, 0, 1, 0,
									0, 0, 1, 0, 0, 0, -1, 0, 0,
									0, -1, 0, 1, 0, 0, 0, 0, 0 };
				for (i = 0; i < 3; i++)
				{
					double ri = i == 0 ? r.x : i == 1 ? r.y : r.z;
					double a0 = -s * ri, a1 = (s - 2 * c1*itheta)*ri, a2 = c1 * itheta;
					double a3 = (c - s * itheta)*ri, a4 = s * itheta;
					for (k = 0; k < 9; k++)
						J[i * 9 + k] = a0 * I[k] + a1 * rrt.val[k] + a2 * drrt[i * 9 + k] + a3 * r_x.val[k] + a4 * d_r_x_[i * 9 + k];
				}
			}
	}
	else if (src->cols == 3 && src->rows == 3)
	{
		Matx33d U, Vt;
		Vec3d W;
		double theta, s, c;
		int step = dst->rows > 1 ? dst->step / elem_size : 1;

		Matx33d R = cvarrToMat(src);

		SVD::compute(R, W, U, Vt);
		R = U * Vt;

		Point3d r(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));

		s = std::sqrt((r.x*r.x + r.y*r.y + r.z*r.z)*0.25);
		c = (R(0, 0) + R(1, 1) + R(2, 2) - 1)*0.5;
		c = c > 1. ? 1. : c < -1. ? -1. : c;
		theta = acos(c);
	    double vth = 1 / (2 * s);
		vth *= theta;
		r *= vth;

		dst->data.db[0] = r.x;
		dst->data.db[step] = r.y;
	}

	if (jacobian)
	{
		if (jacobian->rows == matJ.rows)
			cvCopy(&matJ, jacobian);
		else
			cvTranspose(&matJ, jacobian);
	}

	return 1;
}

void cvTransform2NormalizedPlane(const CvMat* _src, CvMat* _dst, const CvMat* _cameraMatrix)
{
	cvTransform2NormalizedPlaneInternal(_src, _dst, _cameraMatrix);
}

static void cvProjectPoints2Internal(const CvMat* objectPoints,
	const CvMat* r_vec,
	const CvMat* t_vec,
	const CvMat* A,
	const CvMat* distCoeffs,
	CvMat* imagePoints, CvMat* dpdr CV_DEFAULT(NULL),
	CvMat* dpdt CV_DEFAULT(NULL), CvMat* dpdf CV_DEFAULT(NULL),
	CvMat* dpdc CV_DEFAULT(NULL), CvMat* dpdk CV_DEFAULT(NULL))
{
	//matM, 标定板三维世界坐标系下角点坐标
	//_m，图像坐标系下角点坐标
	Ptr<CvMat> matM, _m;

	Ptr<CvMat> _dpdr, _dpdt, _dpdc, _dpdf, _dpdk;
	Ptr<CvMat> _dpdo;

	int i, j, count;
	int calc_derivatives;
	const CvPoint3D64f* M;
	CvPoint2D64f* m;
	//r是Rodrigues表示的旋转，R是3*3的旋转矩阵，dRdr是R的元素对r的Jacobian，所以是27维
	//注意：按照数学惯用表示法dRdr应该是9*3的矩阵，但opencv的实现没有遵循这个习惯，而是用了它的转置
	//也就是dRdr是3*9的矩阵
	//dR1/dr1 dR2/dr1 dR3/dr1 dR4/dr1 dR5/dr1 dR6/dr1 dR7/dr1 dR8/dr1 dR9/dr1
	//dR1/dr2 dR2/dr2 dR3/dr2 dR4/dr2 dR5/dr2 dR6/dr2 dR7/dr2 dR8/dr2 dR9/dr2
	//dR1/dr3 dR2/dr3 dR3/dr3 dR4/dr3 dR5/dr3 dR6/dr3 dR7/dr3 dR8/dr3 dR9/dr3
	double r[3], R[9], dRdr[27], t[3], a[9], k[14] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0 }, fx, fy, cx, cy;
	Matx33d matTilt = Matx33d::eye();
	Matx33d dMatTiltdTauX(0, 0, 0, 0, 0, 0, 0, -1, 0);
	Matx33d dMatTiltdTauY(0, 0, 0, 0, 0, 0, 1, 0, 0);
	CvMat _r, _t, _a = cvMat(3, 3, CV_64F, a), _k;
	CvMat matR = cvMat(3, 3, CV_64F, R), _dRdr = cvMat(3, 9, CV_64F, dRdr);
	double *dpdr_p = 0, *dpdt_p = 0, *dpdk_p = 0, *dpdf_p = 0, *dpdc_p = 0;
	int dpdr_step = 0, dpdt_step = 0, dpdk_step = 0, dpdf_step = 0, dpdc_step = 0;
	
	count = objectPoints->rows * objectPoints->cols; //角点总个数

	matM.reset(cvCreateMat(objectPoints->rows, objectPoints->cols, CV_MAKETYPE(CV_64F, CV_MAT_CN(objectPoints->type))));
	cvConvert(objectPoints, matM);

	_m.reset(cvCreateMat(imagePoints->rows, imagePoints->cols, CV_MAKETYPE(CV_64F, CV_MAT_CN(imagePoints->type))));
	cvConvert(imagePoints, _m);

	//M包含了角点的世界坐标
	//m是世界坐标系的角点按照当前参数投影到成像平面的坐标，是需要计算的，是一个输出
	M = (CvPoint3D64f*)matM->data.db;
	m = (CvPoint2D64f*)_m->data.db;

	_r = cvMat(r_vec->rows, r_vec->cols, CV_MAKETYPE(CV_64F, CV_MAT_CN(r_vec->type)), r);
	//把输入的Rodrigues旋转向量r_vec复制一份局部拷贝_r
	cvConvert(r_vec, &_r);
	//_r包含了输入的Rodrigues旋转向量
	//转换维旋转矩阵表示matR，同时求得matR的9个元素对_r的Jacobian
	//注意：按照数学惯用表示法dRdr应该是9*3的矩阵，但opencv的实现没有遵循这个习惯，而是用了它的转置
	//也就是_dRdr是3*9的矩阵，opencv自己会保持记号正确
	cvRodrigues2(&_r, &matR, &_dRdr);

	//_t为平移向量
	_t = cvMat(t_vec->rows, t_vec->cols, CV_MAKETYPE(CV_64F, CV_MAT_CN(t_vec->type)), t);
	cvConvert(t_vec, &_t);

	cvConvert(A, &_a);
	fx = a[0]; fy = a[4];
	cx = a[2]; cy = a[5];
	
	//_k为畸变系数向量
	_k = cvMat(distCoeffs->rows, distCoeffs->cols,CV_MAKETYPE(CV_64F, CV_MAT_CN(distCoeffs->type)), k);
	cvConvert(distCoeffs, &_k);
	
	if (dpdr)
	{
		_dpdr.reset(cvCloneMat(dpdr));
		dpdr_p = _dpdr->data.db;
		dpdr_step = _dpdr->step / sizeof(dpdr_p[0]);
	}

	if (dpdt)
	{
		_dpdt.reset(cvCloneMat(dpdt));
		dpdt_p = _dpdt->data.db;
		dpdt_step = _dpdt->step / sizeof(dpdt_p[0]);
	}

	if (dpdf)
	{
		_dpdf.reset(cvCloneMat(dpdf));
		dpdf_p = _dpdf->data.db;
		dpdf_step = _dpdf->step / sizeof(dpdf_p[0]);
	}

	if (dpdc)
	{
		_dpdc.reset(cvCloneMat(dpdc));
		dpdc_p = _dpdc->data.db;
		dpdc_step = _dpdc->step / sizeof(dpdc_p[0]);
	}

	if (dpdk)
	{
		_dpdk.reset(cvCloneMat(dpdk));
		dpdk_p = _dpdk->data.db;
		dpdk_step = _dpdk->step / sizeof(dpdk_p[0]);
	}

	calc_derivatives = dpdr || dpdt || dpdf || dpdc || dpdk;

	//count是总共角点个数
	for (i = 0; i < count; i++)
	{
		//(X, Y, Z)为角点的世界坐标
		//（x,y,z）为角点的相机坐标系下的坐标
		double X = M[i].x, Y = M[i].y, Z = M[i].z;
		double x = R[0] * X + R[1] * Y + R[2] * Z + t[0];
		double y = R[3] * X + R[4] * Y + R[5] * Z + t[1];
		double z = R[6] * X + R[7] * Y + R[8] * Z + t[2];
		double r2, r4, r6, a1, a2, a3, cdist;
		double xd, yd, xd0, yd0, invProj;
		
		Vec2d dXdYd;

		double z0 = z;
		z = z ? 1. / z : 1;//注意：此时的z实际上是相机坐标系下真正z值的倒数了,1/z
		x *= z; y *= z; //此时的x,y为归一化平面坐标

		//在归一化平面上进行distortion
		r2 = x * x + y * y;
		r4 = r2 * r2;
		r6 = r4 * r2;
		a1 = 2 * x*y;
		a2 = r2 + 2 * x*x;
		a3 = r2 + 2 * y*y;
		cdist = 1 + k[0] * r2 + k[1] * r4 + k[4] * r6;
		xd0 = x * cdist + k[2] * a1 + k[3] * a2;
		yd0 = y * cdist + k[2] * a3 + k[3] * a1;

		xd = xd0;
		yd = yd0;
		//m为投影到成像平面的像素坐标
		m[i].x = xd * fx + cx;
		m[i].y = yd * fy + cy;

		if (calc_derivatives)
		{
			if (dpdc_p)
			{
				//投影点像素坐标对主点Jacobian
				dpdc_p[0] = 1; 
				dpdc_p[1] = 0; 
				dpdc_p[dpdc_step] = 0;
				dpdc_p[dpdc_step + 1] = 1;
				dpdc_p += dpdc_step * 2;
			}

			if (dpdf_p)
			{
				//投影点对焦距求Jacobian
				dpdf_p[0] = xd; 
				dpdf_p[1] = 0;
				dpdf_p[dpdf_step] = 0;
				dpdf_p[dpdf_step + 1] = yd;
				
				dpdf_p += dpdf_step * 2;
			}
			
			if (dpdk_p)
			{
				//投影点对畸变系数的Jacobian
				dXdYd = Vec2d(x*r2, y*r2); //xnr^2,ynr^2
				dpdk_p[0] = fx * dXdYd(0);
				dpdk_p[dpdk_step] = fy * dXdYd(1);
				dXdYd = Vec2d(x*r4, y*r4); //xnr^4, ynr^4
				dpdk_p[1] = fx * dXdYd(0);
				dpdk_p[dpdk_step + 1] = fy * dXdYd(1);
				if (_dpdk->cols > 2)
				{
					dXdYd = Vec2d(a1, a3); //2xnyn, r^2+2yn^2
					dpdk_p[2] = fx * dXdYd(0);
					dpdk_p[dpdk_step + 2] = fy * dXdYd(1);
					dXdYd =  Vec2d(a2, a1); //r^2+2xn^2, 2xnyn
					dpdk_p[3] = fx * dXdYd(0);
					dpdk_p[dpdk_step + 3] = fy * dXdYd(1);
					if (_dpdk->cols > 4)
					{
						dXdYd = Vec2d(x*r6, y*r6); //与k3相关的项
						dpdk_p[4] = fx * dXdYd(0);
						dpdk_p[dpdk_step + 4] = fy * dXdYd(1);
					}
				}
				dpdk_p += dpdk_step * 2;
			}

			if (dpdt_p)
			{//投影点对平移向量t的Jacobian
				double dxdt[] = { z, 0, -x * z }, dydt[] = { 0, z, -y * z };
				for (j = 0; j < 3; j++)
				{
					double dr2dt = 2 * x*dxdt[j] + 2 * y*dydt[j];
					double dcdist_dt = k[0] * dr2dt + 2 * k[1] * r2*dr2dt + 3 * k[4] * r4*dr2dt;
					double dicdist2_dt = -1 * (k[5] * dr2dt + 2 * k[6] * r2*dr2dt + 3 * k[7] * r4*dr2dt);
					double da1dt = 2 * (x*dydt[j] + y * dxdt[j]);
					double dmxdt = (dxdt[j] * cdist + x * dcdist_dt + x * cdist*dicdist2_dt +
						k[2] * da1dt + k[3] * (dr2dt + 4 * x*dxdt[j]) + k[8] * dr2dt + 2 * r2*k[9] * dr2dt);
					double dmydt = (dydt[j] * cdist + y * dcdist_dt + y * cdist*dicdist2_dt +
						k[2] * (dr2dt + 4 * y*dydt[j]) + k[3] * da1dt + k[10] * dr2dt + 2 * r2*k[11] * dr2dt);
					dXdYd = Vec2d(dmxdt, dmydt);
					dpdt_p[j] = fx * dXdYd(0);
					dpdt_p[dpdt_step + j] = fy * dXdYd(1);
				}
				dpdt_p += dpdt_step * 2;
			}

			if (dpdr_p)
			{//投影点对由axis-angle表示的旋转向量的Jacobian
				double dx0dr[] =
				{
					X*dRdr[0] + Y * dRdr[1] + Z * dRdr[2],
					X*dRdr[9] + Y * dRdr[10] + Z * dRdr[11],
					X*dRdr[18] + Y * dRdr[19] + Z * dRdr[20]
				};
				double dy0dr[] =
				{
					X*dRdr[3] + Y * dRdr[4] + Z * dRdr[5],
					X*dRdr[12] + Y * dRdr[13] + Z * dRdr[14],
					X*dRdr[21] + Y * dRdr[22] + Z * dRdr[23]
				};
				double dz0dr[] =
				{
					X*dRdr[6] + Y * dRdr[7] + Z * dRdr[8],
					X*dRdr[15] + Y * dRdr[16] + Z * dRdr[17],
					X*dRdr[24] + Y * dRdr[25] + Z * dRdr[26]
				};
				for (j = 0; j < 3; j++)
				{
					//注意：这里的z是该点在相机坐标系坐标深度值的倒数
					//在j=0时，dxdr是归一化坐标xn对轴角第1维导数，dydr是归一化坐标yn对轴角第1维导数
					//在j=1时，dxdr是归一化坐标xn对轴角第2维导数，dydr是归一化坐标yn对轴角第2维导数
					//在j=2时，dxdr是归一化坐标xn对轴角第3维导数，dydr是归一化坐标yn对轴角第3维导数
					double dxdr = z * (dx0dr[j] - x * dz0dr[j]); 
					double dydr = z * (dy0dr[j] - y * dz0dr[j]);
					double dr2dr = 2 * x*dxdr + 2 * y*dydr;
					double dcdist_dr = (k[0] + 2 * k[1] * r2 + 3 * k[4] * r4)*dr2dr;
					double da1dr = 2 * (x*dydr + y * dxdr);
					double dmxdr = (dxdr*cdist + x * dcdist_dr + k[2] * da1dr + k[3] * (dr2dr + 4 * x*dxdr) );
					double dmydr = (dydr*cdist + y * dcdist_dr + k[2] * (dr2dr + 4 * y*dydr) + k[3] * da1dr );
					dXdYd = Vec2d(dmxdr, dmydr);
					dpdr_p[j] = fx * dXdYd(0);
					dpdr_p[dpdr_step + j] = fy * dXdYd(1);
				}
				dpdr_p += dpdr_step * 2;
			}
		}
	}

	if (_m != imagePoints)
		cvConvert(_m, imagePoints);

	if (_dpdr != dpdr)
		cvConvert(_dpdr, dpdr);

	if (_dpdt != dpdt)
		cvConvert(_dpdt, dpdt);

	if (_dpdf != dpdf)
		cvConvert(_dpdf, dpdf);

	if (_dpdc != dpdc)
		cvConvert(_dpdc, dpdc);

	if (_dpdk != dpdk)
		cvConvert(_dpdk, dpdk);
}

void cvProjectPoints2(const CvMat* objectPoints,
	const CvMat* r_vec,
	const CvMat* t_vec,
	const CvMat* A,
	const CvMat* distCoeffs,
	CvMat* imagePoints, CvMat* dpdr CV_DEFAULT(NULL),
	CvMat* dpdt CV_DEFAULT(NULL), CvMat* dpdf CV_DEFAULT(NULL),
	CvMat* dpdc CV_DEFAULT(NULL), CvMat* dpdk CV_DEFAULT(NULL))
{
	cvProjectPoints2Internal(objectPoints, r_vec, t_vec, A, distCoeffs, imagePoints, dpdr, dpdt, dpdf, dpdc, dpdk);
}

//此函数根据标定板的角点世界坐标（标定板坐标系下）和对应的图像坐标估计出标定板位姿
void cvFindExtrinsicCameraParams2(const CvMat* objectPoints, const CvMat* imagePoints, const CvMat* A,
	const CvMat* distCoeffs, CvMat* rvec, CvMat* tvec)
{
	const int max_iter = 20;
	Ptr<CvMat> matM, _Mxy, _m, _mn, matL;

	int i, count;
	double a[9], ar[9] = { 1,0,0,0,1,0,0,0,1 }, R[9];
	
	cv::Scalar Mc;
	double param[6]; //外参6个参数，3个代表旋转轴角，3个代表平移
	CvMat matA = cvMat(3, 3, CV_64F, a); //matA为3*3内参矩阵
	CvMat _r = cvMat(3, 1, CV_64F, param);
	CvMat _t = cvMat(3, 1, CV_64F, param + 3);
	CvMat _Mc = cvMat(1, 3, CV_64F, Mc.val);
	CvMat _param = cvMat(6, 1, CV_64F, param);
	CvMat _dpdr, _dpdt;

	count = MAX(objectPoints->cols, objectPoints->rows);
	matM.reset(cvCreateMat(1, count, CV_64FC3));
	_m.reset(cvCreateMat(1, count, CV_64FC2));

	//objectPoints要有一个局部拷贝，为matM
	cv::Mat src = cv::cvarrToMat(objectPoints), dst = cv::cvarrToMat(matM);
	src.copyTo(dst);
	
	//imagePoints要有一个局部考虑，为_m
	src = cv::cvarrToMat(imagePoints), dst = cv::cvarrToMat(_m);
	src.copyTo(dst);
	
	//matA为内参矩阵
	cvConvert(A, &matA);
	// normalize image points to normalized image plane
	//把图像坐标系下的点_m根据内参矩阵matA转换到归一化平面坐标系，结果为_mn
	_mn.reset(cvCreateMat(1, count, CV_64FC2));
	cvTransform2NormalizedPlane(_m, _mn, &matA);

	Mc = cvAvg(matM); //Mc，标定板坐标平均值

	// initialize extrinsic parameters
	double tt[3], h[9], h1_norm, h2_norm;
	CvMat T_transform = cvMat(3, 1, CV_64F, tt);
	CvMat matH = cvMat(3, 3, CV_64F, h); //存储从标定板平面到归一化成像平面的单应矩阵
	CvMat _h1, _h2, _h3;
	
	//标定板坐标系原点移到标定板中心，T_transform是一个3*1的平移向量
	cvTranspose(&_Mc, &T_transform);

	//_Mxy存储了标定板经过去中心化之后的平面坐标
    //标定板坐标系的原点设为所有点的平均值
	_Mxy.reset(cvCreateMat(1, count, CV_64FC2));
	//把原始标定板上的点的坐标重新计算为以坐标均值为原点的坐标系上的坐标，也就是进行了坐标平移
	for (i = 0; i < count; i++)
	{
		const double* Tp = T_transform.data.db;
		const double* src = matM->data.db + i * 3;
		double* dst = _Mxy->data.db + i * 2;
		dst[0] = src[0] + Tp[0];
		dst[1] = src[1] + Tp[1];
	}

	//计算标定板平面和归一化图像平面之间的Homography.
	//这里的实现与课堂内容稍有区别，本质上还是一样的。这里计算了board平面到归一化平面的homography，后面就不用考虑K了
	cvFindHomography(_Mxy, _mn, &matH);
	//以下代码实际上是对H的一种遍历方式。先取到第1个元素列的地址，作为第一列首地址，然后再计算到第2、3列地址
	/*      H0  H1  H2
	 matH = H3  H4  H5
			H6  H7  H8
	*/
	cvGetCol(&matH, &_h1, 0);
	_h2 = _h1; _h2.data.db++;
	_h3 = _h2; _h3.data.db++;
	h1_norm = std::sqrt(h[0] * h[0] + h[3] * h[3] + h[6] * h[6]);
	h2_norm = std::sqrt(h[1] * h[1] + h[4] * h[4] + h[7] * h[7]);

	cvScale(&_h1, &_h1, 1. / MAX(h1_norm, DBL_EPSILON)); //把h1归一化，这就是r1
    cvScale(&_h2, &_h2, 1. / MAX(h2_norm, DBL_EPSILON)); //把h2归一化，这就是r2
	cvScale(&_h3, &_t, 2. / MAX(h1_norm + h2_norm, DBL_EPSILON)); //这里得到了t
	cvCrossProduct(&_h1, &_h2, &_h3); //r1 cross product r2, 得到r3
	//此时的matH里面存储的3列已经成了标定板的旋转矩阵r1,r2,r3
	//得到旋转的轴角表示
	cvRodrigues2(&matH, &_r);
	//我们上面得到的_t，对于标定板来说是坐标中心移到均值之后的平移量。对于原始标定板坐标来说，还要稍加变换一下
	//R(P+T_transformP) + _t = RP+RT_transform+_t,把RT_transform+_t也吸到整个平移向量中(P为原始标定板坐标)
	//_t = matH * T_transform + _t 
	cvMatMulAdd(&matH, &T_transform, &_t, &_t);
	   
	cvReshape(matM, matM, 3, 1);
	cvReshape(_mn, _mn, 2, 1);

	//refine extrinsic parameters using iterative algorithm
	//6是待优化的变量个数，我们这里优化外参，3个旋转、3个平移
	//第2个参数是约束的个数，或者说是方程的个数，由于我们对每个角点都可以列2个方程，所以方程约束个数为2*count
	linLevMarq solver(6, count * 2, cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, max_iter, FLT_EPSILON), true);
	cvCopy(&_param, solver.param);

	for (;;)
	{
		CvMat *matJ = 0, *_err = 0;
		const CvMat *__param = 0;
		
		//update这个函数更新__param
		bool proceed = solver.update(__param, matJ, _err);
		//这里要注意：_param是和param[6]关联起来的，_param更新后，param也就更新了；
		//而旋转和平移参数_r、_t是和param关联的，也就是更新了_r、_t的值
		cvCopy(__param, &_param);
		if (!proceed || !_err)
			break;
		cvReshape(_err, _err, 2, 1);
		if (matJ) //matJ是在solver里面初始化好的，如果matJ不为空，说明需要已经初始化好，需要计算它了
		{
			cvGetCols(matJ, &_dpdr, 0, 3);
			cvGetCols(matJ, &_dpdt, 3, 6);
			//注意：这里的_errs存储的是按照当前参数计算出的角点投影图像坐标
			//同时，这个函数也计算了投影点坐标对旋转向量、平移向量的Jacobian
			cvProjectPoints2(matM, &_r, &_t, &matA, distCoeffs, _err, &_dpdr, &_dpdt, 0, 0, 0);
		}
		else //不需要计算Jacobian了，现在LM需要判断是否收敛，根据当前参数先计算角点投影坐标
		{
			cvProjectPoints2(matM, &_r, &_t, &matA, distCoeffs, _err, 0, 0, 0, 0, 0);
		}
		//计算投影误差，_err经过上面的计算实际上存储的是根据当前外参和内参，标定板角点投影到成像平面上的像素坐标
		//然后用投影坐标-观测坐标，得到误差项
		cvSub(_err, _m, _err);
		cvReshape(_err, _err, 1, 2 * count);
	}
	cvCopy(solver.param, &_param);

	_r = cvMat(rvec->rows, rvec->cols, CV_MAKETYPE(CV_64F, CV_MAT_CN(rvec->type)), param);
	_t = cvMat(tvec->rows, tvec->cols, CV_MAKETYPE(CV_64F, CV_MAT_CN(tvec->type)), param + 3);

	cvConvert(&_r, rvec);
	cvConvert(&_t, tvec);
}

void subMatrix(const Mat& src, Mat& dst, const std::vector<uchar>& cols, const std::vector<uchar>& rows)
{
	CV_Assert(src.channels() == 1);

	int nonzeros_cols = cv::countNonZero(cols);
	Mat tmp(src.rows, nonzeros_cols, CV_64F);

	for (int i = 0, j = 0; i < (int)cols.size(); i++)
	{
		if (cols[i])
		{
			src.col(i).copyTo(tmp.col(j++));
		}
	}

	int nonzeros_rows = cv::countNonZero(rows);
	dst.create(nonzeros_rows, nonzeros_cols, CV_64F);
	for (int i = 0, j = 0; i < (int)rows.size(); i++)
	{
		if (rows[i])
		{
			tmp.row(i).copyTo(dst.row(j++));
		}
	}
}

//这个函数就是把所有标定板的物理点（3D坐标）放在了objPtMat中，
//把所有标定板的图像坐标放在了imgPtMat1中
//npoints是个1维矩阵，与标定板图像个数相同，每个位置了存储了对应标定上的交点个数
static void collectCalibrationData(InputArrayOfArrays objectPoints,InputArrayOfArrays imagePoints1,	Mat& objPtMat, Mat& imgPtMat1, Mat& npoints)
{
	int nimages = (int)objectPoints.total(); //标定板图像个数
	int total = 0;

	for (int i = 0; i < nimages; i++)//统计所有标定图像上角点总个数
	{
		int numberOfObjectPoints = objectPoints.getMat(i).checkVector(3, CV_32F);
		total += numberOfObjectPoints;
	}

	npoints.create(1, (int)nimages, CV_32S);
	objPtMat.create(1, (int)total, CV_32FC3);
	imgPtMat1.create(1, (int)total, CV_32FC2);
	
	Point3f* objPtData = objPtMat.ptr<Point3f>();
	Point2f* imgPtData1 = imgPtMat1.ptr<Point2f>();

	for (int i = 0, j = 0; i < nimages; i++)
	{
		Mat objpt = objectPoints.getMat(i);
		Mat imgpt1 = imagePoints1.getMat(i);
		int numberOfObjectPoints = objpt.checkVector(3, CV_32F);
		npoints.at<int>(i) = numberOfObjectPoints;
		for (int n = 0; n < numberOfObjectPoints; ++n)
		{
			objPtData[j + n] = objpt.ptr<Point3f>()[n];
			imgPtData1[j + n] = imgpt1.ptr<Point2f>()[n];
		}

		j += numberOfObjectPoints;
	}
}

//该函数完成内参与外参数标定任务
//objectPoints，所有标定板图像对应的标定板角点世界坐标，在标定板三维坐标系下定义
//imagePoints，所有标定板图像上的角点图像坐标
//npoints，一维向量，长度为标定板图像个数，每个element的值为对应图像上的角点个数
static double cvCalibrateCamera2Internal(const CvMat* objectPoints,
	const CvMat* imagePoints, const CvMat* npoints,
	CvSize imageSize, CvMat* cameraMatrix, CvMat* distCoeffs,
	CvMat* rvecs, CvMat* tvecs, int flags, CvTermCriteria termCrit)
{
	int NINTRINSIC = 9; //最多内参数个数，我们这个例子是9个
	double reprojErr = 0;

	Matx33d A; //A是3*3内参矩阵
	double k[5] = { 0 }; //存储畸变系数，我们这里只考虑5个
	CvMat matA = cvMat(3, 3, CV_64F, A.val), _k;
	int i, nimages, maxPoints = 0, ni = 0, pos, total = 0, nparams;

	// 0. check the parameters & allocate buffers
	nimages = npoints->rows*npoints->cols;

	//total存储的是总的角点个数，maxPoints是最大的单张标定板图像上的角点个数
	for (i = 0; i < nimages; i++)
	{
		ni = npoints->data.i[i];
		maxPoints = MAX(maxPoints, ni);
		total += ni;
	}

	Mat matM(1, total, CV_64FC3);
	Mat _m(1, total, CV_64FC2);
	//每个物理点根据投影参数都可以投影到像素点，再和对应像素位置计算投影误差，所有allErrors存储了
	//所有物理点的投影误差
	Mat allErrors(1, total, CV_64FC2);

	//matM存储物点坐标
	//_m存储像点坐标
	cvarrToMat(objectPoints).convertTo(matM, CV_64F);
	cvarrToMat(imagePoints).convertTo(_m, CV_64F);
	
	nparams = NINTRINSIC + nimages * 6; //总计待优化的参数个数；每个标定板图像引入6个额外外参，3个旋转、3个平移
	
	//_k现在存储畸变系数数据
	_k = cvMat(distCoeffs->rows, distCoeffs->cols, CV_MAKETYPE(CV_64F, CV_MAT_CN(distCoeffs->type)), k);

	// 1. initialize intrinsic parameters & LM solver
	CvMat _matM = cvMat(matM), m = cvMat(_m); //_matM是所有标定板角点物理坐标，m是对应的图像坐标
	//初始估计内参矩阵，结果存放在matA中，matA为初始化的内参矩阵
	cvInitIntrinsicParams2D(&_matM, &m, npoints, imageSize, &matA);
	
	linLevMarq solver(nparams, 0, termCrit);

	Mat _Ji(maxPoints * 2, NINTRINSIC, CV_64FC1, Scalar(0));
	Mat _Je(maxPoints * 2, 6, CV_64FC1);
	Mat _err(maxPoints * 2, 1, CV_64FC1);

	double* param = solver.param->data.db;

	param[0] = A(0, 0); param[1] = A(1, 1); param[2] = A(0, 2); param[3] = A(1, 2);
	std::copy(k, k + 5, param + 4); //把5个畸变系数拷贝给param完成所有参数初始化

	// 2. initialize extrinsic parameters
	for (i = 0, pos = 0; i < nimages; i++, pos += ni)
	{
		CvMat _ri, _ti;
		ni = npoints->data.i[i];

		//solver中存储参数的地方是param，这是个列向量，包括了与所有图像对应的全部待优化向量
		//这里把和图像i相关的外参数部分关联到引用_ri，_ti，实现参数同步更新
		cvGetRows(solver.param, &_ri, NINTRINSIC + i * 6, NINTRINSIC + i * 6 + 3);
		cvGetRows(solver.param, &_ti, NINTRINSIC + i * 6 + 3, NINTRINSIC + i * 6 + 6);

		//_Mi，第i个标定板图像角点对应的物坐标
		//_mi，第i个标定板图像上的角点的图像坐标
		CvMat _Mi = cvMat(matM.colRange(pos, pos + ni));
		CvMat _mi = cvMat(_m.colRange(pos, pos + ni));

		cvFindExtrinsicCameraParams2(&_Mi, &_mi, &matA, &_k, &_ri, &_ti);
	}

	// 3. run the optimization
	for (;;)
	{
		const CvMat* _param = 0;
		CvMat *_JtJ = 0, *_JtErr = 0; //_JtJ就是J的转置*J, _JtErr就是我们公式中J的转置*f
		double* _errNorm = 0;
		bool proceed = solver.updateAlt(_param, _JtJ, _JtErr, _errNorm);
		double *param = solver.param->data.db, *pparam = solver.prevParam->data.db;
		bool calcJ = solver.state == linLevMarq::CALC_J;
		
		A(0, 0) = param[0]; A(1, 1) = param[1]; A(0, 2) = param[2]; A(1, 2) = param[3];
		std::copy(param + 4, param + 4 + 5, k);

		if (!proceed)
			break;

		reprojErr = 0;

		//总的优化变量是intrinsic (9个) + 6 * imgs, 所有总的Jacobian matrix是一个rows = 2*npoints, cols = 279的大矩阵
		//每张图像都会贡献给这个矩阵一部分信息.
		//注意：我们并不是先计算整个的J，然后再计算JtJ以及JtF；而是在遍历图像的过程中，每次直接更新结果JtJ以及JtF
		for (i = 0, pos = 0; i < nimages; i++, pos += ni)
		{
			CvMat _ri, _ti;
			//ni是当前图像上角点个数，实际上在我们这个例子中每张图像上的角点个数都一样
			ni = npoints->data.i[i];

			//solver中的param是个列向量，包括了全部待优化向量，在我们这个例子中有279个element(9+45*6)
			//这里是把和图像i相关的旋转向量参数关联到_ri,把平移向量参数关联到_ti
			cvGetRows(solver.param, &_ri, NINTRINSIC + i * 6, NINTRINSIC + i * 6 + 3);
			cvGetRows(solver.param, &_ti, NINTRINSIC + i * 6 + 3, NINTRINSIC + i * 6 + 6);

			//_Mi,当前标定板图像对应的物点坐标
			CvMat _Mi = cvMat(matM.colRange(pos, pos + ni));
			//_mi,当前标定板图像上角点的图像坐标
			CvMat _mi = cvMat(_m.colRange(pos, pos + ni));
			//与当前图像相关联的点的投影误差
			CvMat _me = cvMat(allErrors.colRange(pos, pos + ni));

			//_Je,投影点对外参的Jacobian矩阵
			//_Ji，投影点对内参的Jacobian矩阵
			//_err，当前图像每个点在(x,y)投影误差，每个点的误差是2维的
			_Je.resize(ni * 2); 
			_Ji.resize(ni * 2); 
			_err.resize(ni * 2);

			CvMat _mp = cvMat(_err.reshape(2, 1));

			if (calcJ)
			{
				//_dpdr，投影点对rodrigues向量的Jacobian
				//_dpdt，投影点对平移向量的Jacobian
				//_dpdf,投影点对fx,fy的Jacobian
				//_dpdc，投影点对cx,cy的Jacobian
				//_dpdk，投影点对distortion系数的Jacobian
				CvMat _dpdr = cvMat(_Je.colRange(0, 3));
				CvMat _dpdt = cvMat(_Je.colRange(3, 6));
				CvMat _dpdf = cvMat(_Ji.colRange(0, 2));
				CvMat _dpdc = cvMat(_Ji.colRange(2, 4));
				CvMat _dpdk = cvMat(_Ji.colRange(4, NINTRINSIC));

				cvProjectPoints2Internal(&_Mi, &_ri, &_ti, &matA, &_k, &_mp, &_dpdr, &_dpdt, &_dpdf, &_dpdc, &_dpdk);
			}
			else
				cvProjectPoints2(&_Mi, &_ri, &_ti, &matA, &_k, &_mp);

			//在进行这个减法操作之前，_mp是投影点坐标，减去_mi，_mp就成了2维投影误差
			//注意：_mp是和当前图像像素点误差_err数据关联
			cvSub(&_mp, &_mi, &_mp);

			if (calcJ)
			{
				Mat JtJ(cvarrToMat(_JtJ)), JtErr(cvarrToMat(_JtErr));
				
				//把当前图像的Jacobian信息更新至总的Jacobian中
				//每幅图像引入的信息包括内参部分和外参部分，2部分
				JtJ(Rect(0, 0, NINTRINSIC, NINTRINSIC)) += _Ji.t() * _Ji;
				JtJ(Rect(NINTRINSIC + i * 6, NINTRINSIC + i * 6, 6, 6)) = _Je.t() * _Je;
				JtJ(Rect(NINTRINSIC + i * 6, 0, 6, NINTRINSIC)) = _Ji.t() * _Je;
				JtErr.rowRange(0, NINTRINSIC) += _Ji.t() * _err;
				JtErr.rowRange(NINTRINSIC + i * 6, NINTRINSIC + (i + 1) * 6) = _Je.t() * _err;
			}

			double viewErr = norm(_err, NORM_L2SQR);
			reprojErr += viewErr;
		}

		//reprojErr，当前所有图像总的投影误差
		if (_errNorm)
			*_errNorm = reprojErr;

		if (!proceed)
		{
			break;
		}
	}

	// 4. store the results
	cvConvert(&matA, cameraMatrix);
	cvConvert(&_k, distCoeffs);

	for (i = 0, pos = 0; i < nimages; i++)
	{
		CvMat src, dst;

		src = cvMat(3, 1, CV_64F, solver.param->data.db + NINTRINSIC + i * 6);
		dst = cvMat(3, 1, CV_MAT_DEPTH(rvecs->type), rvecs->rows == 1 ?
					rvecs->data.ptr + i * CV_ELEM_SIZE(rvecs->type) :
					rvecs->data.ptr + rvecs->step*i);
		cvConvert(&src, &dst);

		src = cvMat(3, 1, CV_64F, solver.param->data.db + NINTRINSIC + i * 6 + 3);
		dst = cvMat(3, 1, CV_MAT_DEPTH(tvecs->type), tvecs->rows == 1 ?
				tvecs->data.ptr + i * CV_ELEM_SIZE(tvecs->type) :
				tvecs->data.ptr + tvecs->step*i);
		cvConvert(&src, &dst);
	}

	return std::sqrt(reprojErr / total);
}

//_objectPoints，标定板上的角点坐标，以标定板建立的三维世界坐标系，z=0
//_imagePoints,标定板图像上的角点图像坐标
double lincalibrateCameraRO(InputArrayOfArrays _objectPoints,InputArrayOfArrays _imagePoints, Size imageSize, int iFixedPoint, InputOutputArray _cameraMatrix,
	InputOutputArray _distCoeffs, OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, int flags, TermCriteria criteria)
{
	int rtype = CV_64F;

	//相机内参数矩阵
	Mat cameraMatrix = _cameraMatrix.getMat();
	//畸变系数
	Mat distCoeffs = _distCoeffs.getMat();
	
	int nimages = int(_objectPoints.total());
	Mat objPt, imgPt, npoints, rvecM, tvecM;

	bool rvecs_mat_vec = _rvecs.isMatVector();
	bool tvecs_mat_vec = _tvecs.isMatVector();

    _rvecs.create(nimages, 1, CV_64FC3);
    rvecM = _rvecs.getMat();
	_tvecs.create(nimages, 1, CV_64FC3);
	tvecM = _tvecs.getMat();
	
	//这个函数就是把所有标定板的物理点（3D坐标）放在了objPt中，
    //把所有标定板的图像坐标放在了imgPt中
    //npoints是个1维矩阵，与标定板图像个数相同，每个位置存储了对应标定板上的交点个数
	collectCalibrationData(_objectPoints, _imagePoints, objPt, imgPt, npoints);
	
	int np = npoints.at<int>(0); //第一张标定板图像里面角点个数
	
	CvMat c_objPt = cvMat(objPt), c_imgPt = cvMat(imgPt), c_npoints = cvMat(npoints);
	CvMat c_cameraMatrix = cvMat(cameraMatrix), c_distCoeffs = cvMat(distCoeffs);
	CvMat c_rvecM = cvMat(rvecM), c_tvecM = cvMat(tvecM);

	double reprojErr = cvCalibrateCamera2Internal(&c_objPt, &c_imgPt, &c_npoints, cvSize(imageSize),
		&c_cameraMatrix, &c_distCoeffs,
		&c_rvecM,&c_tvecM, flags, cvTermCriteria(criteria));

	// overly complicated and inefficient rvec/ tvec handling to support vector<Mat>
	for (int i = 0; i < nimages; i++)
	{
		if (rvecs_mat_vec)
		{
			_rvecs.create(3, 1, CV_64F, i, true);
			Mat rv = _rvecs.getMat(i);
			memcpy(rv.ptr(), rvecM.ptr(i), 3 * sizeof(double));
		}
		if (tvecs_mat_vec)
		{
			_tvecs.create(3, 1, CV_64F, i, true);
			Mat tv = _tvecs.getMat(i);
			memcpy(tv.ptr(), tvecM.ptr(i), 3 * sizeof(double));
		}
	}

	cameraMatrix.copyTo(_cameraMatrix);
	distCoeffs.copyTo(_distCoeffs);

	return reprojErr;
}

int main()
{
	/*a text file used to store calibration results*/
	ofstream fout("caliberation_result.txt");

	//number of corners of the calibration board
	cv::Size iPatternSize(9, 6);

	//temporarily store corners in 1 calibration board image.
	vector<Point2f> cornersPerImg;

	//Load images in this directory.
	string dirName = "C:\\lin\\计算机视觉导论\\code\\chapter-10-相机标定\\monoCalib\\data";
	vector<cv::String> fileNames;
	//Load all files
	cv::glob(dirName, fileNames); //put all the file names under the folder dirName into vectors fileNames
	cout << "Load images" << endl;
	
	for (auto aFileName : fileNames) 
	{
		cout << aFileName << endl;
	}
	cout << fileNames.size() << " images have been loaded" << endl;
    
	int count = 0;
	int successImageNum = 0;
	vector<vector<Point2f>>  corners_Seq;
	vector<Mat>  image_Seq;

	for (int i = 0; i != fileNames.size(); i++)
	{
		cout << "Frame " << fileNames.at(i) << "..." << endl;
		
		cv::Mat image = imread(fileNames.at(i));
		
		/* extract corners from this image */
		//corner extraction should be performed on gray-scale images
		cv::Mat imageGray;
		cv::cvtColor(image, imageGray, CV_BGR2GRAY);

		bool patternfound = findChessboardCorners(image, iPatternSize, cornersPerImg, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE + CALIB_CB_FAST_CHECK);
		if (!patternfound)
		{
			cout << "cannot find corners: " << fileNames.at(i) << endl;
			getchar();
			exit(1);
		}
		else
		{
			/*sub-pixel precision */
			cornerSubPix(imageGray, cornersPerImg, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
			/* 绘制检测到的角点并保存 */
			cv::Mat imageTemp = image.clone();
			for (int j = 0; j < cornersPerImg.size(); j++)
			{
				cv::circle(imageTemp, cornersPerImg[j], 10, Scalar(0, 0, 255), 2, 8, 0);
			}

			string imageFileName = fileNames.at(i);
			int pos = imageFileName.find_last_of('\\');
			string cornerImgFileName = imageFileName.substr(pos+1, imageFileName.length()-pos-5) + "_corner.jpg";
			imwrite(cornerImgFileName, imageTemp);
			cout << "Frame corner: " << cornerImgFileName << "...end" << endl;

			count = count + cornersPerImg.size();
			successImageNum = successImageNum + 1;
			corners_Seq.push_back(cornersPerImg);
		}
		image_Seq.push_back(image);
	}
	cout << "corner extration is finished!\n";


	cout << "Calibration stards………………" << endl;
	Size square_size = Size(20, 20); //the physical size of each square
	vector<vector<Point3f>>  object_Points; //coordinates of corners on the calibration board 

	/* populate corner coordinates of all board imgs */
	for (int t = 0; t < successImageNum; t++)
	{
		vector<Point3f> tempPointSet;
		for (int i = 0; i < iPatternSize.height; i++)
		{
			for (int j = 0; j < iPatternSize.width; j++)
			{
				/*for each board, for all the corner points, z=0 */
				Point3f tempPoint;
				tempPoint.x = i * square_size.width;
				tempPoint.y = j * square_size.height;
				tempPoint.z = 0;
				tempPointSet.push_back(tempPoint);
			}
		}
		object_Points.push_back(tempPointSet);
	}
		
	Size image_size = image_Seq[0].size();
	cv::Mat intrinsic_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0));
	cv::Mat distortion_coeffs = Mat(1, 5, CV_32FC1, Scalar::all(0));

	std::vector<cv::Vec3d> rotation_vectors; /* rotation vector from the wcs point to the ccs point */
	std::vector<cv::Vec3d> translation_vectors;/* translation vector from the wcs point to the ccs point */
	
	int flags = 0;

	/*cv::calibrateCamera(object_Points, corners_Seq, image_size, intrinsic_matrix, distortion_coeffs, rotation_vectors, translation_vectors, flags,
		cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 1e-6));*/

	lincalibrateCameraRO(object_Points, corners_Seq, image_size, -1, intrinsic_matrix, distortion_coeffs,
		rotation_vectors, translation_vectors, 0, cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 100, 1e-6));

	cout << "calibration finished!\n";

	/************************************************************************
	accuracy assessment of the calibration results
	*************************************************************************/
	cout << "calibartion accuracy assessment starts………………" << endl;
	double total_err = 0.0;         /* sum of average errors for all the images */
	double err = 0.0;               /* average error for each image */
	vector<Point2f>  image_points2; /****  the new image points according to the parameters of the projection  ****/

	cout << "每幅图像的定标误差：" << endl;
	fout << "每幅图像的定标误差：" << endl << endl;
	for (int i = 0; i < image_Seq.size(); i++)
	{
		vector<Point3f> tempPointSet = object_Points[i];
		
		/**** using extrinsics and intrisics, project physical points to the image planes  ****/
		cv::projectPoints(tempPointSet, rotation_vectors[i], translation_vectors[i], intrinsic_matrix, distortion_coeffs, image_points2);
		
		/* compute the average error of the corner points before and after projection*/
		vector<Point2f> tempImagePoint = corners_Seq[i];
		Mat tempImagePointMat = Mat(1, tempImagePoint.size(), CV_32FC2);
		Mat image_points2Mat = Mat(1, image_points2.size(), CV_32FC2);
		for (size_t i = 0; i != tempImagePoint.size(); i++)
		{
			image_points2Mat.at<Vec2f>(0, i) = Vec2f(image_points2[i].x, image_points2[i].y);
			tempImagePointMat.at<Vec2f>(0, i) = Vec2f(tempImagePoint[i].x, tempImagePoint[i].y);
		}
		err = cv::norm(image_points2Mat, tempImagePointMat, NORM_L2);
		total_err += err /= iPatternSize.width * iPatternSize.height;

		cout << "图像" << fileNames.at(i) << "的平均误差：" << err << "像素" << endl;
		fout << "图像" << fileNames.at(i) << "的平均误差：" << err << "像素" << endl;
	}
	cout << "总体平均误差：" << total_err / image_Seq.size() << "像素" << endl;
	fout << "总体平均误差：" << total_err / image_Seq.size() << "像素" << endl << endl;
	cout << "评价完成！" << endl;

	/************************************************************************
	保存定标结果
	*************************************************************************/
	cout << "开始保存定标结果………………" << endl;
	Mat rotation_matrix = Mat(3, 3, CV_32FC1, Scalar::all(0)); /* 保存每幅图像的旋转矩阵 */

	fout << "相机内参数矩阵：" << endl;
	fout << intrinsic_matrix << endl;
	fout << "畸变系数：\n";
	fout << distortion_coeffs << endl;
	for (int i = 0; i < image_Seq.size(); i++)
	{
		fout << "第" << i + 1 << "幅图像的旋转向量：" << endl;
		fout << rotation_vectors[i] << endl;

		/* 将旋转向量转换为相对应的旋转矩阵 */
		cv::Rodrigues(rotation_vectors[i], rotation_matrix);
		fout << "第" << i + 1 << "幅图像的旋转矩阵：" << endl;
		fout << rotation_matrix << endl;
		fout << "第" << i + 1 << "幅图像的平移向量：" << endl;
		fout << translation_vectors[i] << endl;
	}
	cout << "完成保存" << endl;
	fout << endl;
	
	/************************************************************************
	显示定标结果
	*************************************************************************/
	Mat mapx = Mat(image_size, CV_32FC1);
	Mat mapy = Mat(image_size, CV_32FC1);
	Mat R = Mat::eye(3, 3, CV_32F);

	cout << "保存矫正图像" << endl;
	for (int i = 0; i != image_Seq.size(); i++)
	{
		cout << "Frame #" << i + 1 << "..." << endl;
		cv::initUndistortRectifyMap(intrinsic_matrix, distortion_coeffs, R, intrinsic_matrix, image_size, CV_32FC1, mapx, mapy);

		Mat t = image_Seq[i].clone();
		cv::remap(image_Seq[i], t, mapx, mapy, INTER_LINEAR);

		string imageFileName = fileNames.at(i);
		int pos = imageFileName.find_last_of('\\');
		string correctedImgFileName = imageFileName.substr(pos + 1, imageFileName.length() - pos - 5) + "_c.jpg";

		cout << "图像" << correctedImgFileName << "..." << endl;
		imwrite(correctedImgFileName, t);
	}
	cout << "保存结束" << endl;

	return 0;
}


