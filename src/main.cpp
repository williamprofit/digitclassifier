#include <iostream>
#include <Eigen/Dense>

#include <SigmoidFunc.h>
#include <ReLUFunc.h>
#include <Layer.h>
#include <NeuralNetwork.h>

int main()
{
    NeuralNetwork nn;
    Layer l1(5, SigmoidFunc());
    Layer l2(3, SigmoidFunc());

    nn.addLayer(l1);
    nn.addLayer(l2);

    nn.init();

    Eigen::VectorXf input(5);
    input << 1, 2, 3, 4, 5;

    nn.feedforward(input);

    std::cout << nn.getOutput() << std::endl;

    return 0;
}
