#ifndef H_MSE_FUNC
#define H_MSE_FUNC

#include <LossFunc.h>


Eigen::VectorXf MSE(const Eigen::VectorXf& predicted, const Eigen::VectorXf& expected);
Eigen::VectorXf MSEDerivative(const Eigen::VectorXf& predicted, const Eigen::VectorXf& expected);


class MSEFunc : public LossFunc
{
public:
	MSEFunc()
	{
		func = &MSE;
		derivative = &MSEDerivative;
	}
};

#endif
