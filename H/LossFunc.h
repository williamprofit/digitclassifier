#ifndef H_LOSS_FUNC
#define H_LOSS_FUNC

#include <Eigen/Dense>

class LossFunc
{
public:
	Eigen::VectorXf(*func)(const Eigen::VectorXf&, const Eigen::VectorXf&) = nullptr;
	Eigen::VectorXf(*derivative)(const Eigen::VectorXf&, const Eigen::VectorXf&) = nullptr;
};

#endif