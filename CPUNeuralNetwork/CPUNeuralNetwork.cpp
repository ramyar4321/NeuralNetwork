#include "CPUNeuralNetwork.hpp"
#include <iostream>
#include "random"

/**
 * Intialize Neural Network memeber variables.
 */
cpu::NeuralNetwork::NeuralNetwork(unsigned int input_size,
                                  unsigned int layer_p_size,
                                  unsigned int layer_q_size,
                                  unsigned int layer_r_size):
                                  m_input_size(input_size),
                                  m_layer_p_size(layer_p_size),
                                  m_layer_q_size(layer_q_size),
                                  m_layer_r_size(layer_r_size),
                                  m_z1(layer_p_size, 1),
                                  m_z2(layer_q_size, 1),
                                  m_z3(layer_r_size, 1),
                                  m_a1(layer_p_size, 1),
                                  m_a2(layer_q_size, 1),
                                  m_a3(layer_r_size, 1),
                                  // Initialize weights of the neural network to be ones.
                                  // Later on, the weights will be re-initialized using a more 
                                  // sophicticated methode.
                                  m_W1(layer_p_size, std::vector<float>(input_size, 1)),
                                  m_W2(layer_q_size, std::vector<float>(layer_p_size,1)),
                                  m_W3(layer_r_size, std::vector<float>(layer_q_size, 1))
{}

/**
 * Use data to train the neural network.
 */
void cpu::NeuralNetwork::fit(){

    weight_initialization(m_W1, m_input_size, m_layer_p_size);
    weight_initialization(m_W2, m_layer_p_size, m_layer_q_size );
    weight_initialization(m_W3, m_layer_q_size, m_layer_r_size);

    forward_propegation();
}

/**
 * Initialize the weigths of the neural network to random values that come
 * from a Gaussian Distribtuion centered at 0 with standard deviations of 
 * @f$\sqrt{ \farc{1}{n_{I}}} $ where @f$n_{I}$ is the size of layer @f$I$.
 */
void cpu::NeuralNetwork::weight_initialization(std::vector<std::vector<float> > &W, 
                                                const unsigned int &layer_i_size, 
                                                const unsigned int &layer_j_size){
    std::mt19937 generator;
    float mean = 0.0f;
    float stddev = std::sqrt(1 / static_cast<float>(layer_i_size) ); 
    std::normal_distribution<double> normal(mean, stddev);
    for (unsigned int j=0; j<layer_j_size; ++j) {
        for (unsigned int i=0; i<layer_i_size; ++i) {
            W[j][i] = normal(generator);
        }
    } 
}

/**
 * Compute the output of each neuron j in layer J. 
 * The output for each neuron can be computed as follows 
 * @f$z_j = \sum_{i}^I w_{ji} a_i$ where @f$a_i$ is the output of neuron i
 * from the pervious layer I.
 */
void cpu::NeuralNetwork::compute_outputs(std::vector<float> &z,
                                      std::vector<std::vector<float> > &W, std::vector<float> &a,  
                                      const unsigned int &layer_i_size, 
                                      const unsigned int &layer_j_size)
{
    for (unsigned int j=0; j<layer_j_size; ++j) {
        for (unsigned int i=0; i<layer_i_size; ++i) {
            z[j] += W[j][i] * a[i];
        }
    } 
}

/**
 * Compute the activation of each neuron j in layer J using the ReLu activation function. 
 * The activation for each neuron can be computed as follows 
 * @f$z_j = max(0, z_j)$. This method should be called for
 * hidden layers of the neural network.
 */
void cpu::NeuralNetwork::relu_activation(std::vector<float> &a, 
                                         std::vector<float> &z,
                                         const unsigned int &layer_j_size)

{
    for (unsigned int j=0; j<layer_j_size; ++j) {
        if(z[j] > 0.0f ){
            a[j] = z[j];
        }else{
            z[j] = 0.0f;
        }
    } 
}

/**
 * Compute the activation of each neuron j in layer J using the sigmoid activation function. 
 * The activation for each neuron can be computed as follows 
 * @f$z_j = 1/(1+ exp(- z[j]))$. This methode is to be used for the neurons of the 
 * output layer.
 */
void cpu::NeuralNetwork::sigmoid_activation(std::vector<float> &a, 
                                            std::vector<float> &z,
                                            const unsigned int &layer_j_size)
{
    for (unsigned int j=0; j<layer_j_size; ++j) {
        a[j] = 1/(1+ std::exp(- z[j]));
    } 
}