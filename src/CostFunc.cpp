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

	m_cost = VectorXf::Zero(predicted[0].size());
	m_costGradient = VectorXf::Zero(predicted[0].size());

	for (unsigned int i = 0; i < batchSize; i++)
	{
		m_cost += LossFuncTable[m_lossFunc].func(predicted[i], expected[i]);
		m_costGradient += LossFuncTable[m_lossFunc].derivative(predicted[i], expected[i]);
	}

	m_cost /= batchSize;
	m_costGradient /= batchSize;

	m_averageCost = m_cost.sum() / batchSize;
}

VectorXf CostFunc::getCost()
{
	return m_cost;
}

VectorXf CostFunc::getCostGradient()
{
	return m_costGradient;
}

float CostFunc::getAverageCost()
{
	return m_averageCost;
}

void CostFunc::setLossFunc(LossFuncEnum lossFunc)
{
	m_lossFunc = lossFunc;
}