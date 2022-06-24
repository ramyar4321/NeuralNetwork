#include "CPUNeuralNetwork.hpp"

int main(){
    cpu::NeuralNetwork net(3,3,3,1);
    net.fit();

    return 0;
}