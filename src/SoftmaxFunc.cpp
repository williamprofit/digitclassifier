#include <SoftmaxFunc.h>
#include <iostream>

float myExp(float value);

Eigen::VectorXf softmax(const Eigen::VectorXf& activation)
{
	Eigen::VectorXf a(activation.size());

	float sum = activation.unaryExpr(&myExp).sum();

	for (unsigned int i = 0; i < a.size(); i++)
		a(i) = std::exp(activation(i)) / sum;

	return a;
}

Eigen::VectorXf softmaxDerivative(const Eigen::VectorXf& activation)
{
	Eigen::VectorXf a(activation.size());
	Eigen::VectorXf ones(activation.size());
	ones.setOnes();

	a = softmax(activation).cwiseProduct(ones - softmax(activation));

	return a;
}


float myExp(float value)
{
	return std::exp(value);
}