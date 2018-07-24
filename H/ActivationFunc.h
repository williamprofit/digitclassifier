#ifndef H_ACTIVATION_FUNC
#define H_ACTIVATION_FUNC

#include <Eigen/Dense>

class ActivationFunc
{
public:
    Eigen::VectorXf (*func)(const Eigen::VectorXf&) = nullptr;
    Eigen::VectorXf (*derivative)(const Eigen::VectorXf&) = nullptr;
};

#endif
