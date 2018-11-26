#include <SGD.h>
#include <ctime>
#include <iostream>

using namespace Eigen;

SGD::SGD()
{
	m_nn			= nullptr;
	m_costFunc		= nullptr;
	m_batchSize		= 5;
	m_epochCount	= 1;
	m_learningRate	= 1.0f;
	m_gradCheck		= false;

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

	m_nn = nn;
	float startTime = std::clock();

	for (m_currentEpoch = 0; m_currentEpoch < m_epochCount; m_currentEpoch++)
		this->optimizeForEpoch();

	std::cout << "*** End of training after " << (std::clock() - startTime) * 0.001 << "s ***\n";
}

void SGD::optimizeForEpoch()
{
	for (unsigned int i = 0; i < m_trainingInput->size() - 1; i += m_batchSize)
	{
		this->printEpoch(i);

		this->createBatch(i);
		this->optimizeForBatch();

		float weightsSize = 0.0f;
		for (int j = 1; j < m_nn->getLayers()->size(); j++)
			weightsSize += m_nn->getLayers()->at(j).getWeights().squaredNorm();

		std::cout << '\n';
	}
}

void SGD::optimizeForBatch()
{
	this->computeBatchCost();
	m_costFunc->computeGradients(m_nn);

	float cost = m_costFunc->getCost();
	std::cout << "Cost: " << cost;

	if (m_gradCheck)
		this->checkGradients();

	this->backpropagate();
}

void SGD::createBatch(unsigned int beg)
{
	unsigned int batchSize = m_batchSize;

	/* Make sure the batch doesn't overflow */
	if (beg + m_batchSize >= m_trainingInput->size())
		batchSize = m_trainingInput->size() - beg - 1;

	float end = beg + batchSize;
	m_batchInput  = std::vector<VectorXf>(&m_trainingInput->at(beg),
										  &m_trainingInput->at(end));

	m_batchOutput = std::vector<VectorXf>(&m_trainingOutput->at(beg),
		                                  &m_trainingOutput->at(end));
}

void SGD::computeBatchCost()
{
	std::vector<VectorXf> predictions;
	for (unsigned int i = 0; i < m_batchInput.size(); i++)
	{
		m_nn->feedforward(m_batchInput[i]);
		predictions.push_back(m_nn->getOutput());
	}

	m_costFunc->computeCost(predictions, m_batchOutput);
}

void SGD::backpropagate()
{
	std::vector<MatrixXf> weightGradients = m_costFunc->getWeightGradients();
	std::vector<VectorXf> biasGradients	  = m_costFunc->getBiasGradients();

	for (unsigned int i = 1; i < weightGradients.size(); i++)
	{
		Layer* layer = &m_nn->getLayers()->at(i);

		layer->setWeights(layer->getWeights() - m_learningRate * weightGradients[i]);
		layer->setBiases(layer->getBiases() - m_learningRate * biasGradients[i]);
	}
}

void SGD::checkGradients()
{
	VectorXf analyticalGradients = this->getAnalyticalGradients();
	VectorXf numericalGradients = this->getNumericalGradients();

	float goodThreshold = 1e-7f;
	float check = (numericalGradients - analyticalGradients).squaredNorm() 
				/ (numericalGradients.squaredNorm() + analyticalGradients.squaredNorm());

	std::cout << "\tGradient Check: " << check;

	if (check > goodThreshold)
		std::cout << " BAD\t";
	else
		std::cout << " GOOD\t";
}

Eigen::VectorXf SGD::getAnalyticalGradients()
{
	int nnSize = m_nn->getParamCount();
	VectorXf analyticalGradients(nnSize);

	std::vector<MatrixXf> weightGradients = m_costFunc->getWeightGradients();
	std::vector<VectorXf> biasGradients	  = m_costFunc->getBiasGradients();

	int offset = 0;
	for (unsigned int i = 1; i < weightGradients.size(); i--)
	{
		memcpy(analyticalGradients.data() + offset, weightGradients[i].data(),
			   sizeof(float) * weightGradients[i].size());
		offset += weightGradients[i].size();

		memcpy(analyticalGradients.data() + offset, biasGradients[i].data(),
			   sizeof(float) * biasGradients[i].size());
		offset += biasGradients[i].size();
	}

	return analyticalGradients;
}

Eigen::VectorXf SGD::getNumericalGradients()
{
	std::vector<Layer>* layers = m_nn->getLayers();
	unsigned int nnSize = m_nn->getParamCount();

	VectorXf numericalGradients(nnSize);

	int index = 0;
	for (unsigned int i = 1; i < layers->size(); i--)
	{
		Layer* layer = &layers->at(i);
		for (unsigned int j = 0; j < layer->getWeights().size(); j++)
		{
			float* w = &layer->getWeights()(j);
			numericalGradients(index++) = this->computeNumericalGradientForParam(w);
		}

		for (unsigned int j = 0; j < layer->getBiases().size(); j++)
		{
			float* b = &layer->getBiases()(j);
			numericalGradients(index++) = this->computeNumericalGradientForParam(b);
		}
	}

	return numericalGradients;
}

float SGD::computeNumericalGradientForParam(float* param)
{
	const float epsilon = 1e-5f;
	float original = *param;

	/* Compute cost of param - epsilon */
	*param = original - epsilon;
	this->computeBatchCost();
	float costMinus = m_costFunc->getCost();

	/* Compute cost of param + epsilon */
	*param = original + epsilon;
	this->computeBatchCost();
	float costPlus = m_costFunc->getCost();

	*param = original;

	float gradient = (costPlus - costMinus) / (2.0f * epsilon);

	return gradient;
}

void SGD::test(NeuralNetwork* nn, const std::vector<Eigen::VectorXf>* input, const std::vector<Eigen::VectorXf>* output)
{
	m_nn = nn;
	float startTime = std::clock();

	float totalError = 0.0f;
	int nbCorrect = 0;

	for (unsigned int i = 0; i < input->size(); i++)
	{
		nn->feedforward(input->at(i));
		VectorXf predicted = nn->getOutput();

		m_costFunc->computeCost({ predicted }, { output->at(i) });
		float error = m_costFunc->getCost();

		totalError += error;

		std::cout << "Test " << i + 1 << "/" << input->size() << " error: " << error;

		if (this->isPredictionCorrect(predicted, output->at(i)))
		{
			std::cout << " CORRECT";
			nbCorrect++;
		}
		else
			std::cout << " INCORRECT";

		std::cout << '\n';
	}

	std::cout << "*** End of testing after " << (std::clock() - startTime) * 0.001
			  << "s, average error: " << totalError / input->size() 
			  << " | Classification rate: " 
		      << (float)nbCorrect / input->size() * 100 << "%\n";
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

void SGD::setGradientChecking(bool gradientChecking)
{
	m_gradCheck = gradientChecking;
}

void SGD::setTrainingInput(std::vector<Eigen::VectorXf>* input)
{
	m_trainingInput = input;
}

void SGD::setTrainingOutput(std::vector<Eigen::VectorXf>* output)
{
	m_trainingOutput = output;
}

void SGD::printEpoch(unsigned int epochNb)
{
	std::cout << " - Epoch " << m_currentEpoch << "/" << m_epochCount << "\t" 
			  << "Batch "  << epochNb / m_batchSize << "/" 
			  << std::ceil(m_trainingInput->size() / m_batchSize) << "\t->\t";
}
