#include <SigmoidFunc.h>
#include <math.h>

float SigmoidFunc(float activation)
{
    return 1.0f / (1.0f + std::exp(-activation));
}

float SigmoidFuncDerivative(float activation)
{
    return SigmoidFunc(activation) * (1 - SigmoidFunc(activation));
}
