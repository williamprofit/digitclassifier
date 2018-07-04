#include <iostream>
#include <Eigen/Dense>

#include <SigmoidFunc.h>
#include <ReLUFunc.h>

using Eigen::MatrixXd;

int main()
{
    float f = 5.f;
    ActivationFunc af = SigmoidFunc();

    std::cout << af.func(f) << std::endl;

    return 0;
}
