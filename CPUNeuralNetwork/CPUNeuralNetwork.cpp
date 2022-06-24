#include "CPUNeuralNetwork.hpp"
#include <iostream>
#include "random"

// Intialize class memeber variables. 
cpu::NeuralNetwork::NeuralNetwork(unsigned int input_size,
                                  unsigned int layer_p_size,
                                  unsigned int layer_q_size,
                                  unsigned int layer_r_size):
                                  m_input_size(input_size),
                                  m_layer_p_size(layer_p_size),
                                  m_layer_q_size(layer_q_size),
                                  m_layer_r_size(layer_r_size),
                                  // Initialize weights of the neural network to be ones.
                                  // Later on, the weights will be re-initialized using a more 
                                  // sophicticated methode.
                                  m_W1(layer_p_size, std::vector<float>(input_size, 1)),
                                  m_W2(layer_q_size, std::vector<float>(layer_p_size,1)),
                                  m_W3(layer_r_size, std::vector<float>(layer_q_size, 1))
{}

void cpu::NeuralNetwork::weight_initialization(std::vector<std::vector<float> > &W, 
                                                const unsigned int &layer_i_size, 
                                                const unsigned int &layer_j_size){
    std::mt19937 generator;
    float mean = 0.0;
    float stddev = std::sqrt(1 / static_cast<float>(layer_i_size) ); 
    std::normal_distribution<double> normal(mean, stddev);
    for (unsigned int j=0; j<layer_j_size; ++j) {
        for (unsigned int i=0; i<layer_i_size; ++i) {
            W[j][i] = normal(generator);
            std::cout << W[j][i] << std::endl; 
        }
    } 
}

void cpu::NeuralNetwork::fit(){

    weight_initialization(m_W1, m_input_size, m_layer_p_size);
    weight_initialization(m_W2,m_layer_p_size, m_layer_q_size );
    weight_initialization(m_W3, m_layer_q_size, m_layer_r_size);

    forward_propegation();
}

void cpu::NeuralNetwork::forward_propegation(){
    
}