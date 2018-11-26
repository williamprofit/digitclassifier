#include <CostFunc.h>

using namespace Eigen;

CostFunc::CostFunc()
{
}

CostFunc::CostFunc(LossFuncEnum lossFunc) : CostFunc()
{
	this->setLossFunc(lossFunc);
}

void CostFunc::computeCost(const std::vector<VectorXf>& predicted, const std::vector<VectorXf>& expected)
{
	assert(predicted.size() == expected.size() && predicted.size() > 0);

	unsigned int batchSize = predicted.size();

	m_cost = 0.0f;
	m_costGradient = VectorXf::Zero(predicted[0].size());

	for (unsigned int i = 0; i < batchSize; i++)
	{
		m_cost += this->computeCostForPrediction(predicted[i], expected[i]);
		m_costGradient += this->computeCostGradientForPrediction(predicted[i], expected[i]);
	}

	m_cost /= batchSize;
	m_costGradient /= batchSize;
}

void CostFunc::computeGradients(NeuralNetwork* nn)
{
	std::vector<Layer>* layers = nn->getLayers();

	m_weightGradients.resize(layers->size());
	m_biasGradients.resize(layers->size());

	VectorXf daIn = m_costGradient;
	for (unsigned int i = layers->size() - 1; i > 0; i--)
	{
		Layer* layer = &layers->at(i);

		VectorXf dz = ActivationFuncTable[layer->getActivationFunc()].derivative(layer->getIntegration()).cwiseProduct(daIn);

		m_weightGradients[i] = dz * layer->getPrevLayer()->getActivation().transpose();
		m_biasGradients[i] = dz;

		daIn = layer->getWeights().transpose() * dz;
	}
}

float CostFunc::computeCostForPrediction(const VectorXf& predicted, const VectorXf& expected)
{
	float cost = 0.0f;
	for (unsigned int i = 0; i < predicted.size(); i++)
		cost += LossFuncTable[m_lossFunc].func(predicted(i), expected(i));

	return cost;
}

VectorXf CostFunc::computeCostGradientForPrediction(const VectorXf& predicted, const VectorXf& expected)
{
	VectorXf gradient(predicted.size());
	for (unsigned int i = 0; i < predicted.size(); i++)
		gradient(i) = LossFuncTable[m_lossFunc].derivative(predicted(i), expected(i));

	return gradient;
}

float& CostFunc::getCost()
{
	return m_cost;
}

VectorXf& CostFunc::getCostGradient()
{
	return m_costGradient;
}

std::vector<Eigen::VectorXf>& CostFunc::getBiasGradients()
{
	return m_biasGradients;
}

std::vector<Eigen::MatrixXf>& CostFunc::getWeightGradients()
{
	return m_weightGradients;
}

void CostFunc::setLossFunc(LossFuncEnum lossFunc)
{
	m_lossFunc = lossFunc;
}