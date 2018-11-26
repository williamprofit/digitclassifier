#include <SigmoidFunc.h>
#include <cmath>

float singleSigmoid(float x);
float singleSigmoidDerivative(float x);

Eigen::VectorXf sigmoid(const Eigen::VectorXf& activation)
{
	Eigen::VectorXf v = activation.unaryExpr(&singleSigmoid);

	return v;
}

Eigen::VectorXf sigmoidDerivative(const Eigen::VectorXf& activation)
{
	Eigen::VectorXf v = activation.unaryExpr(&singleSigmoidDerivative);

	return v;
}


float singleSigmoid(float x)
{
	return 1.0f / (1.0f + std::exp(-x));
}

float singleSigmoidDerivative(float x)
{
	return singleSigmoid(x) * (1 - singleSigmoid(x));
}