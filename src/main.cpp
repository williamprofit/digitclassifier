#include <DigitDrawer.h>

#include <NeuralNetwork.h>
#include <MNISTLoader.h>
#include <Optimizer.h>
#include <SGD.h>

#include <iostream>
#include <algorithm>

void simpleNetwork();
void train();
void drawer();

int main()
{
	//simpleNetwork();
	//train();
	drawer();

    return 0;
}

void drawer()
{
	DigitDrawer dd(280, 280, true);
	dd.run();
}

void train()
{
	MNISTLoader mnist;
	mnist.load("../res/MNIST");

	NeuralNetwork nn;
	Layer l1(784, ActivationFuncEnum::ACT_NONE);
	Layer l2(16, ActivationFuncEnum::ACT_SIGMOID);
	Layer l3(10, ActivationFuncEnum::ACT_SIGMOID);

	nn.addLayer(l1);
	nn.addLayer(l2);
	nn.addLayer(l3);
	nn.init(true);

	CostFunc costFunc(LossFuncEnum::LOS_MSE);

	SGD optimizer;
	optimizer.setEpochCount(10);
	optimizer.setBatchSize(10);
	optimizer.setCostFunc(&costFunc);
	optimizer.setLearningRate(3.0f);
	optimizer.setTrainingInput(&mnist.getTrainImages());
	optimizer.setTrainingOutput(&mnist.getTrainLabels());
	optimizer.setGradientChecking(false);

	optimizer.optimize(&nn);
	optimizer.test(&nn, &mnist.getTestImages(), &mnist.getTestLabels());

	nn.save("../res/NNsaves/default");
}

void simpleNetwork()
{
	Eigen::VectorXf inputModel(4);
	inputModel << 0.1f, 0.5f, 0.5f, 0.8f;
	Eigen::VectorXf outputModel(2);
	outputModel << 1.0f, 0.0f;

	std::vector<Eigen::VectorXf> train_x(60000, inputModel);
	std::vector<Eigen::VectorXf> train_y(60000, outputModel);
	std::vector<Eigen::VectorXf> test_x(10000, inputModel);
	std::vector<Eigen::VectorXf> test_y(10000, outputModel);

	NeuralNetwork nn;
	Layer l1(4, ActivationFuncEnum::ACT_NONE);
	Layer l2(10, ActivationFuncEnum::ACT_SIGMOID);
	Layer l3(2, ActivationFuncEnum::ACT_SIGMOID);

	nn.addLayer(l1);
	nn.addLayer(l2);
	nn.addLayer(l3);
	nn.init();

	CostFunc costFunc(LossFuncEnum::LOS_MSE);

	SGD optimizer;
	optimizer.setEpochCount(1);
	optimizer.setBatchSize(20);
	optimizer.setCostFunc(&costFunc);
	optimizer.setLearningRate(3.0f);
	optimizer.setTrainingInput(&train_x);
	optimizer.setTrainingOutput(&train_y);
	optimizer.setGradientChecking(true);

	optimizer.optimize(&nn);
	optimizer.test(&nn, &test_x, &test_y);
}