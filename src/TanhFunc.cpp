#include <TanhFunc.h>

float tanhDerivative(float activation)
{
	return 1.0f - std::pow(std::tanh(activation), 2);
}