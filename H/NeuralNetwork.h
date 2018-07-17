#ifndef H_NEURAL_NETWORK
#define H_NEURAL_NETWORK

#include <TrainingInfo.h>
#include <Layer.h>

class NeuralNetwork
{
public:
    NeuralNetwork();
    virtual ~NeuralNetwork();

    virtual void addLayer(const Layer& layer);

    virtual void init();
    virtual void feedforward(const Eigen::VectorXf& input);
	virtual void train(const TrainingInfo& trainingInfo);

    virtual Eigen::VectorXf getOutput();

protected:
	virtual void trainForEpoch(const TrainingInfo& trainingInfo);
	virtual void trainForBatch(const TrainingInfo& trainingInfo, const std::vector<Eigen::VectorXf> batchInput, const std::vector<Eigen::VectorXf> batchExpected);
	virtual void backpropagateError(float learningRate, const Eigen::VectorXf& errorGradient);

	virtual Eigen::VectorXf computeError(const Eigen::VectorXf& output, const Eigen::VectorXf& expected, LossFunc lossFunc);
	virtual Eigen::VectorXf computeErrorGradient(const Eigen::VectorXf& output, const Eigen::VectorXf& expected, LossFunc lossFunc);

    std::vector<Layer> m_layers;
};

#endif
