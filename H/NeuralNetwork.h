#ifndef H_NEURAL_NETWORK
#define H_NEURAL_NETWORK

#include <Layer.h>
#include <string>
#include <vector>

class NeuralNetwork
{
public:
    NeuralNetwork();
    virtual ~NeuralNetwork();

    virtual void addLayer(const Layer& layer);

    virtual void init();
    virtual void feedforward(const Eigen::VectorXf& input);

	virtual bool load(std::string path);
	virtual bool save(std::string path);

    virtual Eigen::VectorXf getOutput();

	virtual std::vector<Layer>* getLayers();

protected:
	virtual void loadLayer(std::ifstream& stream);
	virtual void saveLayer(std::ofstream& stream, Layer* layer);

    std::vector<Layer> m_layers;
};

#endif
