#include <SGD.h>
#include <ctime>
#include <iostream>

using namespace Eigen;

SGD::SGD()
{
	m_costFunc		= nullptr;
	m_batchSize		= 5;
	m_epochCount	= 1;
	m_learningRate	= 1.0f;

	m_trainingInput	 = nullptr;
	m_trainingOutput = nullptr;
}

SGD::~SGD()
{
	m_costFunc		 = nullptr;
	m_trainingInput  = nullptr;
	m_trainingOutput = nullptr;
}

void SGD::optimize(NeuralNetwork* nn)
{
	assert(m_trainingInput->size() == m_trainingOutput->size());

	float startTime = std::clock();

	for (m_currentEpoch = 0; m_currentEpoch < m_epochCount; m_currentEpoch++)
		this->optimizeForEpoch(nn);

	std::cout << "*** End of training after " << (std::clock() - startTime) * 0.001 << "s ***\n";
}

void SGD::optimizeForEpoch(NeuralNetwork* nn)
{
	for (unsigned int i = 0; i < m_trainingInput->size() - 1; i += m_batchSize)
	{
		std::cout << " - Epoch " << m_currentEpoch << "/" << m_epochCount << " " << "- Batch " << i / m_batchSize << "/" << std::ceil(m_trainingInput->size() / m_batchSize) << " | ";

		/* Make sure the batch doesn't overflow */
		int trueBatchSize = m_batchSize;
		if (i + m_batchSize >= m_trainingInput->size())
			m_batchSize = m_trainingInput->size() - i - 1;

		std::vector<VectorXf> batchInput(&m_trainingInput->at(i), &m_trainingInput->at(i + m_batchSize));
		std::vector<VectorXf> batchOutput(&m_trainingOutput->at(i), &m_trainingOutput->at(i + m_batchSize));

		this->optimizeForBatch(nn, &batchInput, &batchOutput);

		m_batchSize = trueBatchSize;
	}
}

void SGD::optimizeForBatch(NeuralNetwork* nn, const std::vector<VectorXf>* batchInput, const std::vector<VectorXf>* batchOutput)
{
	unsigned int batchSize = batchInput->size();
	std::vector<VectorXf> predictions;

	for (unsigned int i = 0; i < batchSize; i++)
	{
		nn->feedforward(batchInput->at(i));
		predictions.push_back(nn->getOutput());
	}

	m_costFunc->computeCost(predictions, *batchOutput);

	VectorXf costGradient = m_costFunc->getCostGradient();
	float averageCost = m_costFunc->getAverageCost();

	std::cout << "Avg Error: " << averageCost << '\n';

	this->backpropagate(nn, costGradient);
}

void SGD::backpropagate(NeuralNetwork* nn, const VectorXf& costGradient)
{
	std::vector<Layer>* layers = nn->getLayers();

	VectorXf daIn = costGradient;
	for (unsigned int i = layers->size() - 1; i > 0; i--)
	{
		Layer* layer = &layers->at(i);

		VectorXf dz = ActivationFuncTable[layer->getActivationFunc()].derivative(layer->getIntegration()).cwiseProduct(daIn);
		daIn = layer->getWeights().transpose() * dz;

		MatrixXf weightsGradient = dz * layers->at(i-1).getActivation().transpose();
		VectorXf biasesGradient  = dz;

		layer->setWeights(layers->at(i).getWeights() - m_learningRate * weightsGradient);
		layer->setBiases(layers->at(i).getBiases() - m_learningRate * biasesGradient);
	}
}

void SGD::test(NeuralNetwork* nn, const std::vector<Eigen::VectorXf>* input, const std::vector<Eigen::VectorXf>* output)
{
	float startTime = std::clock();

	float avgError;
	VectorXf totalError = VectorXf::Zero(output->at(0).size());

	int nbCorrect = 0;

	for (unsigned int i = 0; i < input->size(); i++)
	{
		nn->feedforward(input->at(i));
		VectorXf predicted = nn->getOutput();

		m_costFunc->computeCost({ predicted }, { output->at(i) });
		VectorXf error = m_costFunc->getCost();

		totalError += error;

		std::cout << "Test " << i + 1 << "/" << input->size() << " error: " << error.sum() / error.size();

		if (this->isPredictionCorrect(predicted, output->at(i)))
		{
			std::cout << " CORRECT";
			nbCorrect++;
		}
		else
			std::cout << " INCORRECT";

		std::cout << '\n';
	}

	avgError = (totalError.sum() / input->size()) / totalError.size();

	std::cout << "*** End of testing after " << (std::clock() - startTime) * 0.001 << "s, average error: " << avgError << " | Classification rate: " << (float)nbCorrect / input->size() * 100 << "%\n";
}

bool SGD::isPredictionCorrect(const VectorXf& prediction, const VectorXf& expected)
{
	for (int i = 0; i < prediction.size(); i++)
	{
		if (prediction(i) == prediction.maxCoeff() && expected(i) == expected.maxCoeff())
			return true;
	}

	return false;
}

void SGD::setCostFunc(CostFunc* costFunc)
{
	m_costFunc = costFunc;
}

void SGD::setBatchSize(unsigned int batchSize)
{
	m_batchSize = batchSize;
}

void SGD::setEpochCount(unsigned int epochCount)
{
	m_epochCount = epochCount;
}

void SGD::setLearningRate(float learningRate)
{
	m_learningRate = learningRate;
}

void SGD::setTrainingInput(std::vector<Eigen::VectorXf>* input)
{
	m_trainingInput = input;
}

void SGD::setTrainingOutput(std::vector<Eigen::VectorXf>* output)
{
	m_trainingOutput = output;
}
