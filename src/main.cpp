#include <DigitDrawer.h>

#include <NeuralNetwork.h>
#include <MNISTLoader.h>
#include <Optimizer.h>
#include <SGD.h>

#include <iostream>

void train();
void drawer();

int main()
{
	train();
	drawer();

    return 0;
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
	nn.init();

	CostFunc costFunc(LossFuncEnum::LOS_MSE);

	SGD optimizer;
	optimizer.setEpochCount(30);
	optimizer.setBatchSize(10);
	optimizer.setCostFunc(&costFunc);
	optimizer.setLearningRate(3.0f);
	optimizer.setTrainingInput(&mnist.getTrainImages());
	optimizer.setTrainingOutput(&mnist.getTrainLabels());

	optimizer.optimize(&nn);
	optimizer.test(&nn, &mnist.getTestImages(), &mnist.getTestLabels());

	nn.save("../res/NNsaves/default");
}

void drawer()
{
	DigitDrawer dd(280, 280, true);
	dd.run();
}