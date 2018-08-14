#ifndef H_OPTIMIZER
#define H_OPTIMIZER

#include <NeuralNetwork.h>

class Optimizer
{
public:
	virtual void optimize(NeuralNetwork* nn) = 0;
};

#endif