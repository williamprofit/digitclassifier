#ifndef H_NEURAL_NETWORK
#define H_NEURAL_NETWORK

#include <Layer.h>
#include <vector>

class NeuralNetwork
{
public:
    NeuralNetwork();
    virtual ~NeuralNetwork();

    virtual void addLayer(const Layer& layer);

    virtual void init();
    virtual void feedforward(const Eigen::VectorXf& input);

    virtual Eigen::VectorXf getOutput();

protected:
    std::vector<Layer> m_layers;
};

#endif
