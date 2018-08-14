#include <NeuralNetwork.h>
#include <iostream>
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
	if (m_layers.size() < 2)
	{
		std::cout << "Error: invalid neural network (size < 2), can't feedforward\n";
		return;
	}

    m_layers[0].setActivation(input);

    for (unsigned int i = 1; i < m_layers.size(); i++)
        m_layers[i].fire();
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

std::vector<Layer>* NeuralNetwork::getLayers()
{
	return &m_layers;
}
