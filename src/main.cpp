#include <DigitDrawer.h>

#include <NeuralNetwork.h>
#include <MNISTLoader.h>

#include <iostream>

void train();
void drawer();
void testMNIST();

int main()
{
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
	Layer l3(16, ActivationFuncEnum::ACT_SIGMOID);
	Layer l4(10, ActivationFuncEnum::ACT_SIGMOID);

	nn.addLayer(l1);
	nn.addLayer(l2);
	nn.addLayer(l3);
	nn.addLayer(l4);
	nn.init();

	TrainingInfo trainingInfo;
	trainingInfo.learningRate = 3.0f;
	trainingInfo.batchSize = 5;
	trainingInfo.epochCount = 30;
	trainingInfo.input = &mnist.getTrainImages();
	trainingInfo.expected = &mnist.getTrainLabels();
	trainingInfo.lossFunc = LossFuncEnum::LOS_MSE;

	nn.train(trainingInfo);
	nn.test(mnist.getTestImages(), mnist.getTestLabels(), trainingInfo.lossFunc);

	nn.save("../res/NNsaves/default");
}

void drawer()
{
	DigitDrawer dd(280, 280, true);
	dd.run();
}

void testMNIST()
{
	MNISTLoader mnist(false, true);
	mnist.load("../res/MNIST");

	NeuralNetwork nn;
	nn.load("../res/NNsaves/default");

	for (int n = 0; n < 10000; n++)
	{
		nn.feedforward({ mnist.getTestImages()[n] });
		nn.test({ mnist.getTestImages()[n] }, { mnist.getTestLabels()[n] }, LossFuncEnum::LOS_MSE);

		Eigen::VectorXf output = nn.getOutput();
		Eigen::VectorXf expected = mnist.getTestLabels()[n];

		for (int i = 0; i < output.size(); i++)
			std::cout << output(i) << " - " << expected(i) << '\n';

		system("pause");
	}
}