#include <TanhFunc.h>

float singleTanh(float value);
float squareTanh(float value);

Eigen::VectorXf myTanh(const Eigen::VectorXf& activation)
{
	Eigen::VectorXf v = activation.unaryExpr(&singleTanh);

	return v;
}

Eigen::VectorXf tanhDerivative(const Eigen::VectorXf& activation)
{
	//return 1.0f - std::pow(std::tanh(activation), 2);

	Eigen::VectorXf ones(activation.size());
	ones.setOnes();

	Eigen::VectorXf v = ones - myTanh(activation).unaryExpr(&squareTanh);

	return v;
}


float singleTanh(float value)
{
	return std::tanh(value);
}

float squareTanh(float value)
{
	return std::pow(value, 2.0f);
}