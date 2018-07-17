#ifndef H_CROSS_ENTROPY_FUNC
#define H_CROSS_ENTROPY_FUNC

#include <LossFunc.h>

float crossEntropy(float predicted, float expected);
float crossEntropyDerivative(float predicted, float expected);

class CrossEntropyFunc : public LossFunc
{
public:
	CrossEntropyFunc()
	{
		func = &crossEntropy;
		derivative = &crossEntropyDerivative;
	}
};

#endif