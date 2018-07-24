#include <ReLUFunc.h>
#include <algorithm>

float singleReLU(float value);
float singleReLUDerivative(float value);

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


float singleReLU(float value)
{
	return std::max(0.0f, value);
}

float singleReLUDerivative(float value)
{
	if (value > 0.0f)
		return 1.0f;
	else
		return 0.0f;
}