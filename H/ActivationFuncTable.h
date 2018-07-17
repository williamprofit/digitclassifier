#ifndef H_ACTIVATION_FUNC_TABLE
#define H_ACTIVATION_FUNC_TABLE

#include <ActivationFunc.h>
#include <SigmoidFunc.h>
#include <ReLUFunc.h>
#include <TanhFunc.h>

static ActivationFunc ActivationFuncTable[] = 
{
	SigmoidFunc(),
	ReLUFunc(),
	TanhFunc()
};

enum ActivationFuncEnum
{
	ACT_NONE	= -1,
	ACT_SIGMOID = 0,
	ACT_RELU	= 1,
	ACT_TANH	= 2
};

#endif