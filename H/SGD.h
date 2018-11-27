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
	virtual void setGradientChecking(bool gradientChecking);

	virtual void setTrainingInput(std::vector<Eigen::VectorXf>* input);
	virtual void setTrainingOutput(std::vector<Eigen::VectorXf>* output);

protected:
	virtual void optimizeForEpoch();
	virtual void optimizeForBatch();

	virtual void createBatch(unsigned int beg);
	virtual void computeBatchCost();

	virtual void checkGradients();
	virtual Eigen::VectorXf getAnalyticalGradients();
	virtual Eigen::VectorXf getNumericalGradients();
	virtual float computeNumericalGradientForParam(float* param);

	virtual void backpropagate();

	virtual bool isPredictionCorrect(const Eigen::VectorXf& prediction, const Eigen::VectorXf& expected);

	virtual void printEpoch(unsigned int epochNb);


	NeuralNetwork* m_nn;
	CostFunc* m_costFunc;
	unsigned int m_batchSize;
	unsigned int m_epochCount;
	float m_learningRate;
	bool m_gradCheck;

	std::vector<Eigen::VectorXf>* m_trainingInput;
	std::vector<Eigen::VectorXf>* m_trainingOutput;

	std::vector<Eigen::VectorXf> m_batchInput;
	std::vector<Eigen::VectorXf> m_batchOutput;

	unsigned int m_currentEpoch;
};

#endif