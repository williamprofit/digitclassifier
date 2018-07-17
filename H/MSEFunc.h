#ifndef H_MSE_FUNC
#define H_MSE_FUNC

#include <LossFunc.h>

float MSE(float predicted, float expected);
float MSEDerivative(float predicted, float expected);

class MSEFunc : public LossFunc
{
public:
	MSEFunc()
	{
		func = &MSE;
		derivative = &MSEDerivative;
	}
};

#endif
