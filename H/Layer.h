#ifndef H_LAYER
#define H_LAYER

#include <Eigen/Dense>
#include <ActivationFunc.h>

class Layer
{
public:
    Layer();
    Layer(int size, ActivationFunc activationFunc);
    virtual ~Layer();

    virtual void create(int size, ActivationFunc activationFunc);
    virtual void init(Layer* previousLayer);

    virtual void fire();

	virtual void computeGradients(const Eigen::VectorXf& daIn);
	virtual void applyGradients(float learningRate);

    virtual void setActivation(const Eigen::VectorXf& activation);
	virtual void setWeights(const Eigen::MatrixXf& weights);
	virtual void setBiases(const Eigen::VectorXf& biases);

	virtual Eigen::VectorXf& getIntegration();
	virtual Eigen::VectorXf& getActivation();
	virtual Eigen::MatrixXf& getWeights();
	virtual Eigen::VectorXf& getBiases();
	virtual int getSize();

    virtual void printActivation();

protected:
    void computeActivation();
    void applyActivationFunc();

    int m_size;
    Layer* m_prevLayer;

    ActivationFunc m_activationFunc;

	Eigen::VectorXf m_integrations; /* z = WX+b, integration of values sent to node */
    Eigen::VectorXf m_activations;	/* a = g(z), where g is activationFunc */
    Eigen::MatrixXf m_weights;
    Eigen::VectorXf m_biases;

	Eigen::MatrixXf m_weightsGradient;
	Eigen::VectorXf m_biasesGradient;
};

#endif
