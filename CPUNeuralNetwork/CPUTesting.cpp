#include "CPUTesting.hpp"
#include "CPUNeuralNetwork.hpp"
#include <vector>
#include <iostream>

/**
 * This methode is used to test the compute_outputs method of
 * the NeuralNetwork class. The compute_outputs methode will be
 * tested for each layer of the neural netowrk, thus it 
 * will be tested three times.
 */
void cpu::Testing::test_compute_outputs(){

    // Declare and initialize variables that will be used as
    // input the compute_outputs methode.

    unsigned int input_size = 1;
    unsigned int layer_p_size = 2;
    unsigned int layer_q_size = 2;
    unsigned int layer_r_size = 1;

    // Note, the weights and outputs initialized do not correspond to actual
    // neuron weights and outputs from forward and back propegations. 
    // They are random values simply used to test the compute_output methode.

    std::vector<std::vector<float> > W1 = {{1.1f}, {2.2f}};
    std::vector<std::vector<float> > W2 = {{3.3f, 4.4f}, {5.5f, 6.6f}};
    std::vector<std::vector<float> > W3 = {{7.7f, 8.8f}}; 


    // Note, a3 does not need to be used since the output neuron 
    // is in the last layer and does not feed into any other neuron.
    std::vector<float> x = {1.1f};
    std::vector<float> a1 = {2.0, 3.0};
    std::vector<float> a2 = {4.0, 5.0};

    // Declare and initialize the expected output of compute_outputs.

    // z1 = { x * W1_11, x * W1_21} = {1.1 * 1.1, 1.1 * 2.2}
    std::vector<float> expected_z1 = {1.21f, 2.42f};
    // z2 = { {a1_1 * W2_11 + a1_2 * W2_12}, {a1_1 * W2_21 + a1_2 * W2_22}}
    // z2 = { {2.0 * 3.3 + 3.0 * 4.4}, {2.0 * 5.5 + 3.0 * 6.6}}
    std::vector<float> expected_z2 = {19.8f, 30.8f}; 
    // z3 = {{a2_1 * W3_11 + a2_2 * W3_12}} = {{4.0* 7.7 + 5.0 * 8.8}}
    std::vector<float> expected_z3 = {74.8f};

    // Instantiate an instance of the Neural Network class
    cpu::NeuralNetwork net(input_size,layer_p_size,layer_q_size,layer_r_size);

    // Use mock inputs to test if methode produces expected results
    std::vector<float> actual_z1 = net.compute_outputs(W1, x, input_size, layer_p_size);
    std::vector<float> actual_z2 = net.compute_outputs(W2, a1, layer_p_size, layer_q_size);
    std::vector<float> actual_z3 = net.compute_outputs(W3, a2, layer_q_size, layer_r_size);


    if ( std::equal(actual_z1.begin(), actual_z1.end(), expected_z1.begin(),
                        [](double value1, double value2)
                        {
                            constexpr double epsilon = 0.01; 
                            return std::abs(value1 - value2) < epsilon;
                        }))
        std::cout << "Test succeded for z1.\n";
    else
        std::cout << "Test failed for z1.\n";

    if ( std::equal(actual_z2.begin(), actual_z2.end(), expected_z2.begin(),
                        [](double value1, double value2)
                        {
                            constexpr double epsilon = 0.01; 
                            return std::abs(value1 - value2) < epsilon;
                        }))
        std::cout << "Test succeded for z2.\n";
    else
        std::cout << "Test failed for z2.\n";

    if ( std::equal(actual_z3.begin(), actual_z3.end(), expected_z3.begin(),
                        [](double value1, double value2)
                        {
                            constexpr double epsilon = 0.01; 
                            return std::abs(value1 - value2) < epsilon;
                        }))
        std::cout << "Test succeded for z3.\n";
    else
        std::cout << "Test failed for z3.\n";


}