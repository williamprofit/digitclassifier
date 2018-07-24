#ifndef H_CROSS_ENTROPY_FUNC
#define H_CROSS_ENTROPY_FUNC

#include <LossFunc.h>


Eigen::VectorXf crossEntropy(const Eigen::VectorXf& predicted, const Eigen::VectorXf& expected);
Eigen::VectorXf crossEntropyDerivative(const Eigen::VectorXf& predicted, const Eigen::VectorXf& expected);


class CrossEntropyFunc : public LossFunc
{
public:
	CrossEntropyFunc()
	{
		func = &crossEntropy;
		derivative = &crossEntropyDerivative;
	}
};

#endif