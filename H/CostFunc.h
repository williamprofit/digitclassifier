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
	virtual void computeGradients(NeuralNetwork* nn);

	virtual float&			 getCost();
	virtual Eigen::VectorXf& getCostGradient();

	virtual std::vector<Eigen::VectorXf>& getBiasGradients();
	virtual std::vector<Eigen::MatrixXf>& getWeightGradients();

	virtual void setLossFunc(LossFuncEnum lossFunc);

protected:
	virtual float computeCostForPrediction(const Eigen::VectorXf& predicted, const Eigen::VectorXf& expected);
	virtual Eigen::VectorXf computeCostGradientForPrediction(const Eigen::VectorXf& predicted, const Eigen::VectorXf& expected);

	float			m_cost;
	Eigen::VectorXf m_costGradient;

	std::vector<Eigen::VectorXf> m_biasGradients;
	std::vector<Eigen::MatrixXf> m_weightGradients;

	LossFuncEnum m_lossFunc;
};

#endif