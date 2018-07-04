#include <SigmoidFunc.h>
#include <math.h>

float sigmoid(float activation)
{
    return 1.0f / (1.0f + std::exp(-activation));
}

float sigmoidDerivative(float activation)
{
    return sigmoid(activation) * (1 - sigmoid(activation));
}
