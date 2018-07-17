#ifndef H_TANH_FUNC
#define H_TANH_FUNC

#include <ActivationFunc.h>
#include <cmath>

float tanhDerivative(float activation);

class TanhFunc : public ActivationFunc
{
public:
	TanhFunc()
	{
		func = &std::tanh;
		derivative = &tanhDerivative;
	}
};

#endif