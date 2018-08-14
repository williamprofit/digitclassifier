#ifndef H_COST_FUNC
#define H_COST_FUNC

#include <Eigen/Dense>
#include <vector>
#include <LossFuncTable.h>
#include <NeuralNetwork.h>

class CostFunc
{
public:
	CostFunc();
	CostFunc(LossFuncEnum lossFunc);

	virtual void computeCost(const std::vector<Eigen::VectorXf>& predicted, const std::vector<Eigen::VectorXf>& expected);

	virtual Eigen::VectorXf getCost();
	virtual Eigen::VectorXf getCostGradient();
	virtual float			getAverageCost();

	virtual void setLossFunc(LossFuncEnum lossFunc);

protected:
	Eigen::VectorXf m_cost;
	Eigen::VectorXf m_costGradient;
	float			m_averageCost;

	LossFuncEnum m_lossFunc;
};

#endif