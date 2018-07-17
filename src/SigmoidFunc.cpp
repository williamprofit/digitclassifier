#include <SigmoidFunc.h>
#include <cmath>

float sigmoid(float activation)
{
    return 1.0f / (1.0f + std::exp(-activation));
}

float sigmoidDerivative(float activation)
{
    return sigmoid(activation) * (1.0f - sigmoid(activation));
}
