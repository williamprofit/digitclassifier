#ifndef H_SGD
#define H_SGD

#include <Optimizer.h>

#include <CostFunc.h>

class SGD : public Optimizer
{
public:
	SGD();
	virtual ~SGD();

	virtual void optimize(NeuralNetwork* nn);
	virtual void test(NeuralNetwork* nn, const std::vector<Eigen::VectorXf>* input, const std::vector<Eigen::VectorXf>* output);

	virtual void setCostFunc(CostFunc* costFunc);
	virtual void setBatchSize(unsigned int batchSize);
	virtual void setEpochCount(unsigned int epochCount);
	virtual void setLearningRate(float learningRate);

	virtual void setTrainingInput(std::vector<Eigen::VectorXf>* input);
	virtual void setTrainingOutput(std::vector<Eigen::VectorXf>* output);

protected:
	virtual void optimizeForEpoch(NeuralNetwork* nn);
	virtual void optimizeForBatch(NeuralNetwork* nn, const std::vector<Eigen::VectorXf>* batchInput, const std::vector<Eigen::VectorXf>* batchOutput);
	virtual void backpropagate(NeuralNetwork* nn, const Eigen::VectorXf& costGradient);

	virtual bool isPredictionCorrect(const Eigen::VectorXf& prediction, const Eigen::VectorXf& expected);

	CostFunc* m_costFunc;
	unsigned int m_batchSize;
	unsigned int m_epochCount;
	float m_learningRate;

	std::vector<Eigen::VectorXf>* m_trainingInput;
	std::vector<Eigen::VectorXf>* m_trainingOutput;

	unsigned int m_currentEpoch;
};

#endif SGD