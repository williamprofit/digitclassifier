#ifndef H_SIGMOID_FUNC
#define H_SIGMOID_FUNC

#include <ActivationFunc.h>


Eigen::VectorXf sigmoid(const Eigen::VectorXf& activation);
Eigen::VectorXf sigmoidDerivative(const Eigen::VectorXf& activation);


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
