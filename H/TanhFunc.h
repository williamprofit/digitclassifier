#ifndef H_TANH_FUNC
#define H_TANH_FUNC

#include <ActivationFunc.h>
#include <cmath>


Eigen::VectorXf myTanh(const Eigen::VectorXf& activation);
Eigen::VectorXf tanhDerivative(const Eigen::VectorXf& activation);


class TanhFunc : public ActivationFunc
{
public:
	TanhFunc()
	{
		func = &myTanh;
		derivative = &tanhDerivative;
	}
};

#endif