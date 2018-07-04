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

    virtual int getSize();
    virtual Eigen::VectorXf getActivation();

    virtual void setActivation(Eigen::VectorXf activation);

    virtual void printActivation();

protected:
    void computeActivation();
    void applyActivationFunc();

    int m_size;
    Layer* m_prevLayer;

    ActivationFunc m_activationFunc;

    Eigen::VectorXf m_activations;
    Eigen::MatrixXf m_weights;
    Eigen::VectorXf m_biases;
};

#endif
