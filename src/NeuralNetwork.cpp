#include <NeuralNetwork.h>
#include <iostream>
#include <ctime>
#include <fstream>
#include <EigenFileIO.h>

using namespace Eigen;

NeuralNetwork::NeuralNetwork()
{
}

NeuralNetwork::~NeuralNetwork()
{
}

void NeuralNetwork::addLayer(const Layer& layer)
{
    m_layers.push_back(layer);
}

void NeuralNetwork::init()
{
    for (unsigned int i = 0; i < m_layers.size(); i++)
    {
        Layer* prevLayer = nullptr;
        if (i > 0)
            prevLayer = &m_layers[i-1];

        m_layers[i].init(prevLayer);
    }
}

void NeuralNetwork::feedforward(const VectorXf& input)
{
    if (m_layers.size() == 0) return;

    m_layers[0].setActivation(input);

    for (unsigned int i = 1; i < m_layers.size(); i++)
        m_layers[i].fire();
}

void NeuralNetwork::train(const TrainingInfo& trainingInfo)
{
	float startTime = std::clock();

	for (unsigned int epoch = 0; epoch < trainingInfo.epochCount; epoch++)
	{
		std::cout << "~ Epoch " << epoch << " ~\n";
		this->trainForEpoch(trainingInfo);
	}

	std::cout << "*** End of training after " << (std::clock() - startTime) * 0.001 << "s ***\n";
}

void NeuralNetwork::trainForEpoch(const TrainingInfo& trainingInfo)
{
	unsigned int batchSize = trainingInfo.batchSize;

	for (unsigned int i = 0; i < trainingInfo.input->size() - 1; i += batchSize)
	{
		std::cout << "- Batch " << i / trainingInfo.batchSize << "/" << std::ceil(trainingInfo.input->size() / trainingInfo.batchSize) << " -\n";
		/* Make sure we don't overflow */
		if (i + trainingInfo.batchSize >= trainingInfo.input->size())
			batchSize = trainingInfo.input->size() - i - 1;

		/* Create batches */
		std::vector<VectorXf> batchInput(&trainingInfo.input->at(i), &trainingInfo.input->at(i + batchSize));
		std::vector<VectorXf> batchExpected(&trainingInfo.expected->at(i), &trainingInfo.expected->at(i + batchSize));

		this->trainForBatch(trainingInfo, batchInput, batchExpected);
	}
}

void NeuralNetwork::trainForBatch(const TrainingInfo& trainingInfo, const std::vector<Eigen::VectorXf> batchInput, const std::vector<Eigen::VectorXf> batchExpected)
{
	VectorXf batchError = VectorXf::Zero(this->getOutput().size());
	VectorXf batchErrorGradient = VectorXf::Zero(this->getOutput().size());

	for (unsigned int i = 0; i < batchInput.size(); i++)
	{
		this->feedforward(batchInput[i]);
		VectorXf output = this->getOutput();

		VectorXf exampleError = this->computeError(output, batchExpected[i], trainingInfo.lossFunc);
		VectorXf exampleErrorGradient = this->computeErrorGradient(output, batchExpected[i], trainingInfo.lossFunc);

		batchError += exampleError;
		batchErrorGradient += exampleErrorGradient;
	}

	/* Average out error vectors over batch size */
	batchError /= trainingInfo.batchSize;
	batchErrorGradient /= trainingInfo.batchSize;

	/* Compute average error over all outputs */
	float avgError = batchError.sum() / batchError.count();
	std::cout << "Avg Error:\n" << avgError << '\n';

	this->backpropagateError(trainingInfo.learningRate, batchErrorGradient);
}

void NeuralNetwork::backpropagateError(float learningRate, const VectorXf& errorGradient)
{
	this->m_layers.back().computeGradients(errorGradient);
	this->m_layers.back().applyGradients(learningRate);
}

VectorXf NeuralNetwork::computeError(const VectorXf& output, const VectorXf& expected, LossFuncEnum lossFunc)
{
	VectorXf error = VectorXf::Zero(output.size());

	for (unsigned int i = 0; i < output.size(); i++)
		error[i] = LossFuncTable[lossFunc].func(output[i], expected[i]);

	return error;
}

VectorXf NeuralNetwork::computeErrorGradient(const VectorXf& output, const VectorXf& expected, LossFuncEnum lossFunc)
{
	VectorXf errorGradient = VectorXf::Zero(output.size());

	for (unsigned int i = 0; i < output.size(); i++)
		errorGradient[i] = LossFuncTable[lossFunc].derivative(output[i], expected[i]);

	return errorGradient;
}

float NeuralNetwork::test(const std::vector<VectorXf>& input, const std::vector<VectorXf>& expected, LossFuncEnum lossFunc)
{
	float startTime = std::clock();

	float avgError;
	VectorXf totalError = VectorXf::Zero(expected[0].size());

	for (unsigned int i = 0; i < input.size(); i++)
	{
		this->feedforward(input[i]);
		VectorXf output = this->getOutput();

		VectorXf error = this->computeError(output, expected[i], lossFunc);

		std::cout << "Test " << i+1 << "/" << input.size() << " error: " << error.sum() << "\n";

		totalError += error;
	}

	avgError = totalError.sum() / input.size();

	std::cout << "*** End of testing after " << (std::clock() - startTime) * 0.001 << "s, average error: " << avgError << "\n";

	return avgError;
}

bool NeuralNetwork::load(std::string path)
{
	std::ifstream file(path, std::ios::binary);

	if (!file.is_open())
	{
		std::cout << "Failed to open file " << path << " for loading.\n";
		return false;
	}

	uint32_t layerCount;
	file.read((char*)&layerCount, sizeof(layerCount));

	for (uint32_t i = 0; i < layerCount; i++)
		this->loadLayer(file);

	for (unsigned int i = 0; i < m_layers.size(); i++)
	{
		Layer* prevLayer = nullptr;
		if (i > 0)
			prevLayer = &m_layers[i - 1];

		m_layers[i].setPrevLayer(prevLayer);
	}

	file.close();

	return true;
}

void NeuralNetwork::loadLayer(std::ifstream& stream)
{
	Layer layer;

	uint32_t layerSize;
	int8_t activationFunc;

	stream.read((char*)&layerSize, sizeof(layerSize));
	stream.read((char*)&activationFunc, sizeof(activationFunc));

	layer.create(layerSize, ActivationFuncEnum(activationFunc));

	MatrixXf weights;
	VectorXf biases;

	readMatrixBinary(stream, weights);
	readVectorBinary(stream, biases);

	layer.setWeights(weights);
	layer.setBiases(biases);

	this->addLayer(layer);
}

bool NeuralNetwork::save(std::string path)
{
	std::ofstream file(path, std::ios::binary | std::ios::trunc);

	if (!file.is_open())
	{
		std::cout << "Failed to create file " << path << " for saving.\n";
		return false;
	}

	uint32_t layerCount = (uint32_t)m_layers.size();
	file.write((char*)&layerCount, sizeof(layerCount));
	
	for (uint32_t i = 0; i < layerCount; i++)
		this->saveLayer(file, &m_layers[i]);

	file.close();

	return true;
}

void NeuralNetwork::saveLayer(std::ofstream& stream, Layer* layer)
{
	uint32_t layerSize = (uint32_t)layer->getSize();
	int8_t	activationFunc = (int8_t)layer->getActivationFunc();

	stream.write((char*)&layerSize, sizeof(layerSize));
	stream.write((char*)&activationFunc, sizeof(activationFunc));

	writeMatrixBinary(stream, layer->getWeights());
	writeVectorBinary(stream, layer->getBiases());
}

VectorXf NeuralNetwork::getOutput()
{
    if (m_layers.size() == 0)
        return VectorXf::Zero(1);

    return m_layers.back().getActivation();
}
