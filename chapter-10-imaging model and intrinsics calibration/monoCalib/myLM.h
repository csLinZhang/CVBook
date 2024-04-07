#ifndef LIN_LM_H
#define LIN_LM_H
//////////////////////////////////////////////////////////////////////////////////////////
#include "opencv2/core/types_c.h"

class linLevMarq
{
public:
	linLevMarq();
	linLevMarq(int nparams, int nerrs, CvTermCriteria criteria =
		cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, DBL_EPSILON),
		bool completeSymmFlag = false);
	~linLevMarq();
	void init(int nparams, int nerrs, CvTermCriteria criteria =
		cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, DBL_EPSILON),
		bool completeSymmFlag = false);
	bool update(const CvMat*& param, CvMat*& J, CvMat*& err);
	bool updateAlt(const CvMat*& param, CvMat*& JtJ, CvMat*& JtErr, double*& errNorm);

	void clear();
	void step();
	enum { DONE = 0, STARTED = 1, CALC_J = 2, CHECK_ERR = 3 };

	//cv::Ptr<CvMat> mask;
	cv::Ptr<CvMat> prevParam;
	cv::Ptr<CvMat> param;
	cv::Ptr<CvMat> J;
	cv::Ptr<CvMat> err;
	cv::Ptr<CvMat> JtJ;
	cv::Ptr<CvMat> JtJN;
	cv::Ptr<CvMat> JtErr;
	cv::Ptr<CvMat> JtJV;
	cv::Ptr<CvMat> JtJW;
	double prevErrNorm, errNorm;
	int lambdaLg10;
	CvTermCriteria criteria;
	int state;
	int iters;
	bool completeSymmFlag;
	int solveMethod;
};
#endif /* LIN_LM_H */