#include <CrossEntropyFunc.h>

#include <cmath>
#include <limits>


float crossEntropy(float predicted, float expected)
{
	float value = -(expected * std::log(predicted) + (1 - expected) * std::log(1 - predicted));

	return value;
}

float crossEntropyDerivative(float predicted, float expected)
{
	float value = -(expected / predicted) + (1 - expected) / (1 - predicted);

	return value;
}
