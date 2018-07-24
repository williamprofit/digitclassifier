#ifndef H_SOFTMAX_FUNC
#define H_SOFTMAX_FUNC

#include <ActivationFunc.h>


Eigen::VectorXf softmax(const Eigen::VectorXf& activation);
Eigen::VectorXf softmaxDerivative(const Eigen::VectorXf& activation);


class SoftmaxFunc : public ActivationFunc
{
public:
	SoftmaxFunc()
	{
		func = &softmax;
		derivative = &softmaxDerivative;
	}
};

#endif