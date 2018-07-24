#ifndef H_RELU_FUNC
#define H_RELU_FUNC

#include <ActivationFunc.h>


Eigen::VectorXf ReLU(const Eigen::VectorXf& activation);
Eigen::VectorXf ReLUDerivative(const Eigen::VectorXf& activation);


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
