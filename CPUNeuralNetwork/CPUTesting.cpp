#include "CPUTesting.hpp"
#include "CPUNeuralNetwork.hpp"
#include <vector>
#include <iostream>
#include <functional>

/**
 * This methode is used to test the compute_outputs method of
 * the NeuralNetwork class. The compute_outputs methode will be
 * tested for each layer of the neural network, thus it 
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

    std::vector<std::vector<double> > W1 = {{1.1f}, {2.2f}};
    std::vector<std::vector<double> > W2 = {{3.3f, 4.4f}, {5.5f, 6.6f}};
    std::vector<std::vector<double> > W3 = {{7.7f, 8.8f}}; 


    // Note, a3 does not need to be used since the output neuron 
    // is in the last layer and does not feed into any other neuron.
    std::vector<double> x = {1.1f};
    std::vector<double> a1 = {2.0, 3.0};
    std::vector<double> a2 = {4.0, 5.0};

    // Declare and initialize the expected output of compute_outputs.

    // z1 = { x * W1_11, x * W1_21} = {1.1 * 1.1, 1.1 * 2.2}
    std::vector<double> expected_z1 = {1.21f, 2.42f};
    // z2 = { {a1_1 * W2_11 + a1_2 * W2_12}, {a1_1 * W2_21 + a1_2 * W2_22}}
    // z2 = { {2.0 * 3.3 + 3.0 * 4.4}, {2.0 * 5.5 + 3.0 * 6.6}}
    std::vector<double> expected_z2 = {19.8f, 30.8f}; 
    // z3 = {{a2_1 * W3_11 + a2_2 * W3_12}} = {{4.0* 7.7 + 5.0 * 8.8}}
    std::vector<double> expected_z3 = {74.8f};

    // Instantiate an instance of the Neural Network class
    cpu::NeuralNetwork net(input_size,layer_p_size,layer_q_size,layer_r_size);

    // Use mock inputs to test if methode produces expected results

    std::vector<double> actual_z1 = net.compute_outputs(W1, x, input_size, layer_p_size);
    std::vector<double> actual_z2 = net.compute_outputs(W2, a1, layer_p_size, layer_q_size);
    std::vector<double> actual_z3 = net.compute_outputs(W3, a2, layer_q_size, layer_r_size);

    // Function pointer to helper function to be used as callback function
    // when comparing actual and expected values.
    std::function<bool(double,double)> f = &cpu::Testing::areFloatEqual;

    if ( std::equal(actual_z1.begin(), actual_z1.end(), expected_z1.begin(),f))
        std::cout << "Test succeded for z1.\n";
    else
        std::cout << "Test failed for z1.\n";

    if ( std::equal(actual_z2.begin(), actual_z2.end(), expected_z2.begin(),f))
        std::cout << "Test succeded for z2.\n";
    else
        std::cout << "Test failed for z2.\n";

    if ( std::equal(actual_z3.begin(), actual_z3.end(), expected_z3.begin(),f))
        std::cout << "Test succeded for z3.\n";
    else
        std::cout << "Test failed for z3.\n";


}

/**
 * This methode is used to test the relu_activation method of
 * the NeuralNetwork class. 
 */
void cpu::Testing::test_relu_activation(){

    // Declare and initialize variables that will be used as
    // input the relu_activation methode.

    unsigned int input_size = 1;
    unsigned int layer_p_size = 2;
    unsigned int layer_q_size = 2;
    unsigned int layer_r_size = 1;

    std::vector<double> z1 = {2.28f, 4.19f};
    std::vector<double> z2 = {-2.28f, -4.19f};

    // Declare and initialize the expected output of relu_activation.

    // a1 = { max(0, z1_1), max(0, z1_2)} = { max(0, 2.28f), max(0, 4.19f)}
    std::vector<double> expected_a1 = {2.28f, 4.19f};
    // a2 = { max(0, z2_1), max(0, z2_2)} = { max(0, -2.28f), max(0, -4.19f)}
    std::vector<double> expected_a2 = {0.0f, 0.0f}; 

    // Instantiate an instance of the Neural Network class
    cpu::NeuralNetwork net(input_size,layer_p_size,layer_q_size,layer_r_size);

    // Use mock inputs to test if methode produces expected results
    std::vector<double> actual_a1 = net.relu_activation(z1, layer_p_size);
    std::vector<double> actual_a2 = net.relu_activation(z2, layer_q_size);

    // Function pointer to helper function to be used as callback function
    // when comparing actual and expected values.
    std::function<bool(double,double)> f = &cpu::Testing::areFloatEqual;

     if ( std::equal(actual_a1.begin(), actual_a1.end(), expected_a1.begin(), f))
        std::cout << "Test succeded for a1.\n";
    else
        std::cout << "Test failed for a1.\n";

    if ( std::equal(actual_a2.begin(), actual_a2.end(), expected_a2.begin(),f))
        std::cout << "Test succeded for a2.\n";
    else
        std::cout << "Test failed for a2.\n";


}


/**
 * This methode is used to test the sigmoid_activation method of
 * the NeuralNetwork class. Three tests will be conducted. The sigmoid 
 * method will be tested with positive, negative, and zero values.
 */
void cpu::Testing::test_sigmoid_activation(){

    // Declare and initialize variables that will be used as
    // input the sigmoid_activation methode.

    unsigned int input_size = 1;
    unsigned int layer_p_size = 2;
    unsigned int layer_q_size = 2;
    unsigned int layer_r_size = 1;

    double z_1 = 3.3;
    double z_2 = -3.3;
    double z_3 = 0.0;

    // Declare and initialize the expected output of sigmoid_activation.

    // a = { 1/ (e^(-3.33)) ,  1/ (e^(+3.33)),  1/ (e^(0))} 
    // a = 1/ (e^(-3.33))
    double expected_a_1 = 0.965;
    // a = 1/ (e^(-3.33))
    double expected_a_2 = 0.03456;
    // a = 1/ (e^(0))
    double expected_a_3 = 0.5;

    // Instantiate an instance of the Neural Network class
    cpu::NeuralNetwork net(input_size,layer_p_size,layer_q_size,layer_r_size);

    // Use mock inputs to test if methode produces expected results
    double actual_a_1 = net.sigmoid_activation(z_1);
    double actual_a_2 = net.sigmoid_activation(z_2);
    double actual_a_3 = net.sigmoid_activation(z_3);

    // Function pointer to helper function to be used as callback function
    // when comparing actual and expected values.
    std::function<bool(float,float)> f = &cpu::Testing::areFloatEqual;

    if ( areFloatEqual(expected_a_1, actual_a_1))
        std::cout << "Test succeded for a1.\n";
    else
        std::cout << "Test failed for a1.\n";

    if ( areFloatEqual(expected_a_2, actual_a_2))
        std::cout << "Test succeded for a2.\n";
    else
        std::cout << "Test failed for a2.\n";

    if ( areFloatEqual(expected_a_3, actual_a_3))
        std::cout << "Test succeded for a3.\n";
    else
        std::cout << "Test failed for a3.\n";

}

/**
 * This methode is used to test the compute_loss method of
 * the NeuralNetwork class. We want to test if logarithm of 
 * the entropy loss function is defined for when a is zero.
 * Two tests will be conducted. 
 * 1. y = 1 and a = 0 where y is the actual outcome for a given sample of the dataset
 *    and a is the output of the output neuron. This test will determine if the logarithm
 *    of the first term of the entropy loss function is undefined for the choose output a.
 * 2. y = 0 and a = 1 where y is the actual outcome for a given sample of the dataset
 *    and a is the output of the output neuron. This test will determine if the logarithm
 *    of the second term of the entropy loss function is undefined for the choose output a.
 */
void cpu::Testing::test_compute_loss(){

    // Declare and initialize variables that will be used as
    // input the sigmoid_activation methode.

    unsigned int input_size = 1;
    unsigned int layer_p_size = 2;
    unsigned int layer_q_size = 2;
    unsigned int layer_r_size = 1;

    double y_1 = 1.0;
    double a_1 = 0.0;
    double y_2 = 0.0;
    double a_2 = 1.0;

    // Declare and initialize the expected output of sigmoid_activation.

    // loss = -y*log(a + epsilon) - (1-y)*log(1 - a + epsilon)
    // loss = -log(0.0001) 4
    float expected_loss = 9.21; // Expected loss for both tests.

    // Instantiate an instance of the Neural Network class
    cpu::NeuralNetwork net(input_size,layer_p_size,layer_q_size,layer_r_size);

    // Use mock inputs to test if methode produces expected results
    float actual_loss_1 = net.compute_loss(y_1, a_1);
    float actual_loss_2 = net.compute_loss(y_2, a_2);

    if ( areFloatEqual(expected_loss, actual_loss_1))
        std::cout << "First test succeeded for the entropy loss methode.\n";
    else
        std::cout << "Second test failed for the entropy loss methode.\n";
    if ( areFloatEqual(expected_loss, actual_loss_2))
        std::cout << "First test succeeded for the entropy loss methode.\n";
    else
        std::cout << "SecondTest failed for the entropy loss methode.\n";


}


/**
 * Determine if two float values are equal with a fixed error. 
 * Fixed point errors are not used for comparison between floating point values
 * but it will suffice for our usage. 
 */
bool cpu::Testing::areFloatEqual(double a, double b){
    constexpr double epsilon = 0.01; 
    return std::abs(a - b) < epsilon;
}