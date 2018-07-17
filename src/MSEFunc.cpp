#include <MSEFunc.h>
#include <cmath>

float MSE(float predicted, float expected)
{
	return 0.5 * std::pow(predicted - expected, 2);
}

float MSEDerivative(float predicted, float expected)
{
	return 2.0f * (predicted - expected);
}