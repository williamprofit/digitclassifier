#include <NeuralNetwork.h>
#include <iostream>

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

VectorXf NeuralNetwork::getOutput()
{
    if (m_layers.size() == 0)
        return VectorXf::Zero(1);

    return m_layers.back().getActivation();
}
