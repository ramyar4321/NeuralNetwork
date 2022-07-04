#include "CPUNeuralNetwork.hpp"
#include "CPUTesting.hpp"

int main(){
    //cpu::NeuralNetwork net(3,3,3,1);
    //net.fit();

    cpu::Testing test;
    test.test_compute_outputs();
    test.test_relu_activation();
    test.test_sigmoid_activation();
    test.test_compute_loss();
    

    return 0;
}