#ifndef H_LAYER
#define H_LAYER

#include <ActivationFunc.h>

class Layer
{
public:
    Layer();
    Layer(int size, ActivationFunc activationFunc, ActivationFunc activationFuncDerivative);
    virtual ~Layer();

protected:
};

#endif
