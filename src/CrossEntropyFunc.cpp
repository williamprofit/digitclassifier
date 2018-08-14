#include <CrossEntropyFunc.h>
#include <cmath>

float myLog(float);
void nanToNum(Eigen::VectorXf& vec);

Eigen::VectorXf crossEntropy(const Eigen::VectorXf& predicted, const Eigen::VectorXf& expected)
{
	Eigen::VectorXf ones(predicted.size());
	ones.setOnes();

	Eigen::VectorXf error = -(expected.cwiseProduct(predicted.unaryExpr(&myLog)) + (ones - expected).cwiseProduct((ones - predicted).unaryExpr(&myLog)));
	nanToNum(error);

	return error;
}

Eigen::VectorXf crossEntropyDerivative(const Eigen::VectorXf& predicted, const Eigen::VectorXf& expected)
{
	Eigen::VectorXf ones(predicted.size());
	ones.setOnes();

	Eigen::VectorXf derivative = -(expected.cwiseQuotient(predicted)) + (ones - expected).cwiseQuotient(ones - predicted);
	nanToNum(derivative);

	return derivative;
}

void nanToNum(Eigen::VectorXf& vec)
{
	for (unsigned int i = 0; i < vec.size(); i++)
	{
		if (std::isnan(vec(i)))
			vec(i) = 0.0f;

		if (std::isinf(vec(i)))
		{
			if (vec(i) < 0)
				vec(i) = -1000000.0f;
			else
				vec(i) = +1000000.0f;
		}
	}
}

float myLog(float value)
{
	return std::log(value);
}