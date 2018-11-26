#include <TanhFunc.h>

float singleTanh(float x);
float singleTanhDerivative(float x);

Eigen::VectorXf myTanh(const Eigen::VectorXf& activation)
{
	Eigen::VectorXf v = activation.unaryExpr(&singleTanh);

	return v;
}

Eigen::VectorXf tanhDerivative(const Eigen::VectorXf& activation)
{
	Eigen::VectorXf v = activation.unaryExpr(&singleTanhDerivative);

	return v;
}


float singleTanh(float x)
{
	return std::tanh(x);
}

float singleTanhDerivative(float x)
{
	return 1.0f - std::pow(std::tanh(x), 2);
}