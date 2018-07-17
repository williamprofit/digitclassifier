#ifndef H_LOSS_FUNC_TABLE
#define H_LOSS_FUNC_TABLE

#include <LossFunc.h>
#include <MSEFunc.h>
#include <CrossEntropyFunc.h>

static LossFunc LossFuncTable[] =
{
	MSEFunc(),
	CrossEntropyFunc()
};

enum LossFuncEnum
{
	LOS_NONE			= -1,
	LOS_MSE				= 0,
	LOS_CROSS_ENTROPY	= 1
};

#endif