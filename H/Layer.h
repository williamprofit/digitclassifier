#ifndef H_LAYER
#define H_LAYER

#include <Eigen/Dense>
#include <ActivationFuncTable.h>

class Layer
{
public:
    Layer();
    Layer(unsigned int size, ActivationFuncEnum activationFunc);
    virtual ~Layer();

    virtual void create(unsigned int size, ActivationFuncEnum activationFunc);
    virtual void init(Layer* previousLayer);

    virtual void fire();

	virtual void computeGradients(const Eigen::VectorXf& daIn);
	virtual void applyGradients(float learningRate);

    virtual void setActivation(const Eigen::VectorXf& activation);
	virtual void setWeights(const Eigen::MatrixXf& weights);
	virtual void setBiases(const Eigen::VectorXf& biases);
	virtual void setActivationFunc(ActivationFuncEnum activationFunc);
	virtual void setPrevLayer(Layer* prevLayer);

	virtual Eigen::VectorXf& getIntegration();
	virtual Eigen::VectorXf& getActivation();
	virtual Eigen::MatrixXf& getWeights();
	virtual Eigen::VectorXf& getBiases();
	virtual unsigned int getSize();
	virtual ActivationFuncEnum getActivationFunc();

    virtual void printActivation();

protected:
    virtual void computeActivation();
    virtual void applyActivationFunc();

    unsigned int m_size;
    Layer* m_prevLayer;

    ActivationFuncEnum m_activationFunc;

	Eigen::VectorXf m_integrations; /* z = WX+b, integration of values sent to node */
    Eigen::VectorXf m_activations;	/* a = g(z), where g is activationFunc */
    Eigen::MatrixXf m_weights;
    Eigen::VectorXf m_biases;

	Eigen::MatrixXf m_weightsGradient;
	Eigen::VectorXf m_biasesGradient;
};

#endif
