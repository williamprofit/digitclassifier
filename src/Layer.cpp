#include <Layer.h>
#include <iostream>

using namespace Eigen;

Layer::Layer() : m_prevLayer(nullptr)
{
}

Layer::Layer(int size, ActivationFunc activationFunc)
{
    this->create(size, activationFunc);
}

Layer::~Layer()
{
}

void Layer::create(int size, ActivationFunc activationFunc)
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
    m_activations = m_weights * m_prevLayer->getActivation() + m_biases;
    this->applyActivationFunc();
}

void Layer::applyActivationFunc()
{
    m_activations = m_activations.unaryExpr(m_activationFunc.func);
}

int Layer::getSize()
{
    return m_size;
}

VectorXf Layer::getActivation()
{
    return m_activations;
}

void Layer::setActivation(VectorXf activation)
{
    m_activations = activation;
}

void Layer::printActivation()
{
    std::cout << m_activations << std::endl;
}
