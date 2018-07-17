#include <iostream>
#include <Eigen/Dense>

#include <SigmoidFunc.h>
#include <TanhFunc.h>
#include <ReLUFunc.h>
#include <Layer.h>
#include <CrossEntropyFunc.h>
#include <NeuralNetwork.h>
#include <MSEFunc.h>

int main()
{
    NeuralNetwork nn;
    Layer l1(5, ActivationFunc());
    Layer l2(3, SigmoidFunc());

    nn.addLayer(l1);
    nn.addLayer(l2);
    nn.init();


	Eigen::VectorXf inputModel(5);
	inputModel << 0.1, 0.2, 0.3, 0.4, 0.5;
	std::vector<Eigen::VectorXf> input(100, inputModel);

	Eigen::VectorXf outputModel(3);
	outputModel << 0.42, 0.9, 0.42;
	std::vector<Eigen::VectorXf> output(100, outputModel);


	TrainingInfo trainingInfo;
	trainingInfo.learningRate	= 0.1;
	trainingInfo.batchSize		= 5;
	trainingInfo.epochCount		= 10;
	trainingInfo.input			= &input;
	trainingInfo.expected		= &output;
	trainingInfo.lossFunc		= CrossEntropyFunc();

	std::cout << "--- TEST ---" << std::endl;
	nn.feedforward(inputModel);
	std::cout << nn.getOutput() << std::endl;

	
	nn.train(trainingInfo);


	std::cout << "--- TEST ---" << std::endl;
	nn.feedforward(inputModel);
	std::cout << nn.getOutput() << std::endl;

	system("pause");

    return 0;
}