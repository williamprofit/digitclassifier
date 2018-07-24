#include <CrossEntropyFunc.h>
#include <cmath>

float myLog(float);

Eigen::VectorXf crossEntropy(const Eigen::VectorXf& predicted, const Eigen::VectorXf& expected)
{
	Eigen::VectorXf ones(predicted.size());
	ones.setOnes();

	Eigen::VectorXf error = -(expected.cwiseProduct(predicted.unaryExpr(&myLog)) + (ones - expected).cwiseProduct((ones - predicted).unaryExpr(&myLog)));
	
	return error;
}

Eigen::VectorXf crossEntropyDerivative(const Eigen::VectorXf& predicted, const Eigen::VectorXf& expected)
{
	Eigen::VectorXf ones(predicted.size());
	ones.setOnes();

	Eigen::VectorXf derivative = -(expected.cwiseQuotient(predicted)) + (ones - expected).cwiseQuotient(ones - predicted);
	return derivative;
}


float myLog(float value)
{
	return std::log(value);
}