#include <iostream>
#include <Eigen/Dense>

#include <NeuralNetwork.h>
#include <MNISTLoader.h>

int main()
{
	MNISTLoader mnist;
	mnist.load("../res/MNIST");

    NeuralNetwork nn;
    Layer l1(784, ActivationFuncEnum::ACT_NONE);
	Layer l2(15, ActivationFuncEnum::ACT_TANH);
    Layer l3(10, ActivationFuncEnum::ACT_SIGMOID);

    nn.addLayer(l1);
    nn.addLayer(l2);
	nn.addLayer(l3);
    nn.init();

	TrainingInfo trainingInfo;
	trainingInfo.learningRate	= 0.1f;
	trainingInfo.batchSize		= 5;
	trainingInfo.epochCount		= 1;
	trainingInfo.input			= &mnist.getTrainImages();
	trainingInfo.expected		= &mnist.getTrainLabels();
	trainingInfo.lossFunc		= LossFuncEnum::LOS_MSE;

	nn.train(trainingInfo);
	nn.test(mnist.getTestImages(), mnist.getTestLabels(), LossFuncEnum::LOS_MSE);

	nn.save("../res/save1");

	system("pause");

    return 0;
}