#ifndef H_TRAINING_INFO
#define H_TRAINING_INFO

#include <Eigen/Dense>
#include <LossFunc.h>
#include <vector>

struct TrainingInfo
{
	float learningRate;
	unsigned int batchSize;
	unsigned int epochCount;

	std::vector<Eigen::VectorXf>* input;
	std::vector<Eigen::VectorXf>* expected;

	LossFunc lossFunc;
};

#endif