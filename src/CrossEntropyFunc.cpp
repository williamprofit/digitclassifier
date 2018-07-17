#include <CrossEntropyFunc.h>
#include <cmath>

float crossEntropy(float predicted, float expected)
{
	return -(expected * std::log(predicted) + (1.0f - expected) * std::log(1.0f - predicted));
}

float crossEntropyDerivative(float predicted, float expected)
{
	return -(expected / predicted) + (1.0f - expected) / (1.0f - predicted);
}