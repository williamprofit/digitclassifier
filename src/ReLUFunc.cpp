#include <ReLUFunc.h>
#include <algorithm>

float ReLU(float activation)
{
    return std::max(0.0f, activation);
}

float ReLUDerivative(float activation)
{
    if (activation > 0.0f)
        return 1.0f;
    else
        return 0.0f;
}
