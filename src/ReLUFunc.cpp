#include <ReLUFunc.h>
#include <algorithm>

float singleReLU(float x);
float singleReLUDerivative(float x);

Eigen::VectorXf ReLU(const Eigen::VectorXf& activation)
{
	Eigen::VectorXf a = activation.unaryExpr(&singleReLU);

	return a;
}

Eigen::VectorXf ReLUDerivative(const Eigen::VectorXf& activation)
{
	Eigen::VectorXf a = activation.unaryExpr(&singleReLUDerivative);

	return a;
}


float singleReLU(float x)
{
	return std::max(0.0f, x);
}

float singleReLUDerivative(float x)
{
	if (x > 0.0f)
		return 1.0f;
	else
		return 0.0f;
}