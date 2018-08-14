#include <Layer.h>
#include <iostream>

using namespace Eigen;

Layer::Layer() : m_prevLayer(nullptr)
{
}

Layer::Layer(unsigned int size, ActivationFuncEnum activationFunc)
{
    this->create(size, activationFunc);
}

Layer::~Layer()
{
	m_prevLayer = nullptr;
}

void Layer::create(unsigned int size, ActivationFuncEnum activationFunc)
{
    m_size = size;
    m_activationFunc = activationFunc;
}

void Layer::init(Layer* previousLayer)
{
    m_prevLayer = previousLayer;
    m_activations = VectorXf::Zero(m_size);

    if (m_prevLayer != nullptr)
    {
        m_weights = MatrixXf::Random(m_size, m_prevLayer->getSize());
        m_biases  = VectorXf::Random(m_size);
    }
}

void Layer::fire()
{
    if (m_prevLayer == nullptr)
        return;
    else
        this->computeActivation();
}

void Layer::computeActivation()
{
	m_integrations = m_weights * m_prevLayer->getActivation() + m_biases;
	this->applyActivationFunc();
}

void Layer::applyActivationFunc()
{
	m_activations = ActivationFuncTable[m_activationFunc].func(m_integrations);
}

void Layer::setActivation(const VectorXf& activation)
{
	m_activations = activation;
}

void Layer::setWeights(const MatrixXf& weights)
{
	m_weights = weights;
}

void Layer::setBiases(const VectorXf& biases)
{
	m_biases = biases;
}

void Layer::setActivationFunc(ActivationFuncEnum activationFunc)
{
	m_activationFunc = activationFunc;
}

void Layer::setPrevLayer(Layer* prevLayer)
{
	m_prevLayer = prevLayer;
}

unsigned int Layer::getSize()
{
    return m_size;
}

VectorXf& Layer::getIntegration()
{
	return m_integrations;
}

VectorXf& Layer::getActivation()
{
    return m_activations;
}

MatrixXf& Layer::getWeights()
{
	return m_weights;
}

ActivationFuncEnum Layer::getActivationFunc()
{
	return m_activationFunc;
}

VectorXf& Layer::getBiases()
{
	return m_biases;
}

void Layer::printActivation()
{
    std::cout << m_activations << std::endl;
}
