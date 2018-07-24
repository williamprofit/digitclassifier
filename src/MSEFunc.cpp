#include <MSEFunc.h>
#include <cmath>

float squareMSE(float value);

Eigen::VectorXf MSE(const Eigen::VectorXf& predicted, const Eigen::VectorXf& expected)
{
	Eigen::VectorXf error = 0.5f * (predicted - expected).unaryExpr(&squareMSE);

	return error;
}

Eigen::VectorXf MSEDerivative(const Eigen::VectorXf& predicted, const Eigen::VectorXf& expected)
{
	Eigen::VectorXf derivative = predicted - expected;

	return derivative;
}


float squareMSE(float value)
{
	return std::pow(value, 2.0f);
}