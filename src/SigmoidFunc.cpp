#include <SigmoidFunc.h>
#include <cmath>

float singleSigmoid(float value);

Eigen::VectorXf sigmoid(const Eigen::VectorXf& activation)
{
	Eigen::VectorXf a = activation.unaryExpr(&singleSigmoid);

	return a;
}

Eigen::VectorXf sigmoidDerivative(const Eigen::VectorXf& activation)
{
	Eigen::VectorXf ones(activation.size());
	ones.setOnes();

	Eigen::VectorXf a = activation.unaryExpr(&singleSigmoid).cwiseProduct(ones - activation.unaryExpr(&singleSigmoid));

	return a;
}


float singleSigmoid(float value)
{
	return 1.0f / (1.0f + std::exp(-value));
}