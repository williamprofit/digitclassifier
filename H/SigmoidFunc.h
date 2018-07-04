#ifndef H_SIGMOID_FUNC
#define H_SIGMOID_FUNC

#include <ActivationFunc.h>
#include <math.h>


float sigmoid(float activation);
float sigmoidDerivative(float activation);


class SigmoidFunc : public ActivationFunc
{
public:
    SigmoidFunc()
    {
        func = &sigmoid;
        derivative = &sigmoidDerivative;
    }
};

#endif
