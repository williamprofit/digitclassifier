#ifndef H_RELU_FUNC
#define H_RELU_FUNC

#include <ActivationFunc.h>


float ReLU(float activation);
float ReLUDerivative(float activation);


class ReLUFunc : public ActivationFunc
{
public:
    ReLUFunc()
    {
        func = &ReLU;
        derivative = & ReLUDerivative;
    }
};

#endif
