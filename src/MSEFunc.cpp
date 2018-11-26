#include <MSEFunc.h>

#include <cmath>

float MSE(float predicted, float expected)
{
	return 0.5f * std::pow(predicted - expected, 2);
}

float MSEDerivative(float predicted, float expected)
{
	return predicted - expected;
}