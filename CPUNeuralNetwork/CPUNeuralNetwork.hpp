#ifndef CPU_NEURAL_NETWORK
#define CPU_NEURAL_NETWORK

#include <vector>
#include "Matrix.hpp"


namespace cpu {
    /** 
     * Class NeuralNetwork
     * 
     * A three layer artificial neural network. The size of each layer 
     * is specified by the user during instantiation of the Neural
     * Network class.
     * 
     */
    class NeuralNetwork{

        public:
            NeuralNetwork(unsigned int input_size,
                          unsigned int layer_p_size,
                          unsigned int layer_q_size,
                          unsigned int layer_r_size);

            cpu::Matrix weight_initialization( const unsigned int &layer_i_size, 
                                                                    const unsigned int &layer_j_size);

            
            void fit(std::vector<std::vector<double> > &X);

            void forward_propegation();
            
            std::vector<double> compute_outputs(const cpu::Matrix &W, 
                                               const std::vector<double> &a,  
                                               const unsigned int &layer_i_size, 
                                               const unsigned int &layer_j_size);

            std::vector<double> relu_activation(const std::vector<double> &z,
                                                const unsigned int &layer_j_size);

            double sigmoid_activation(const double &z);

            double compute_loss(const double &y, 
                                const double &a);

        private:

            // Number of input neurons.
            unsigned int m_input_size;
            // Number of neurons in first hidden layer.
            unsigned int m_layer_p_size;
            // Number of neurons in second hidden layer.
            unsigned int m_layer_q_size;
            // Number of output neurons.
            unsigned int m_layer_r_size;

            // Store the output of the neurons for each layer, excluding the input layer.
            std::vector<double> m_z1;
            std::vector<double> m_z2;
            // The last layer corresponds to the output layer.
            // The output layer has only one neuron so an std::vector is unnecessary.
            double m_z3;

            // Store the activations of the neurons for each layer, excluding the input layer.
            std::vector<double> m_a1;
            std::vector<double> m_a2;
            // The last layer corresponds to the output layer.
            // The output layer has only one neuron so an std::vector is unnecessary.
            double m_a3;

            // Maxtrix that will store weights between input layer and first hidden layer
            Matrix m_W1;
            // Maxtrix that will store weights between first hidden layer and second hidden layer.
            Matrix m_W2;
            // Maxtrix that will store weights between second hidden layer and output layer.
            Matrix m_W3;



    };
}

#endif // End of CPU_NEURAL_NETWORK