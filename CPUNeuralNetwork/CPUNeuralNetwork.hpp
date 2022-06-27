#ifndef CPU_NEURAL_NETWORK
#define CPU_NEURAL_NETWORK

#include <vector>


namespace cpu {
    /** 
     * Class NeuralNetwork
     * 
     * A three layer artificial neural network where the sizes of each
     * layer are set during instantiating. 
     * 
     */
    class NeuralNetwork{
        public:
            NeuralNetwork(unsigned int input_size,
                                  unsigned int layer_p_size,
                                  unsigned int layer_q_size,
                                  unsigned int layer_r_size);

            void weight_initialization(std::vector<std::vector<float> > &W, 
                                        const unsigned int &layer_i_size, 
                                        const unsigned int &layer_j_size);

            
            void fit();

            void forward_propegation();
            
            void compute_outputs(std::vector<float> &z,
                                      const std::vector<std::vector<float> > &W, const std::vector<float> &a,  
                                      const unsigned int &layer_i_size, 
                                      const unsigned int &layer_j_size);

            void relu_activation(std::vector<float> &a, 
                                         const std::vector<float> &z,
                                         const unsigned int &layer_j_size);

            void sigmoid_activation(std::vector<float> &a, 
                                            const std::vector<float> &z,
                                            const unsigned int &layer_j_size);

            float compute_loss(const std::vector<float> &y, 
                               const std::vector<float> &a,
                               const unsigned int &layer_j_size);

        private:

            // Number of input layer neurons.
            unsigned int m_input_size;
            // Number of neurons in first hidden layer.
            unsigned int m_layer_p_size;
            // Number of neurons in second hidden layer.
            unsigned int m_layer_q_size;
            // Number of neurons in the output layer.
            unsigned int m_layer_r_size;

            // Store the output of the neurons for each layer, excluding the input layer.
            std::vector<float> m_z1;
            std::vector<float> m_z2;
            std::vector<float> m_z3;

            // Store the activations of the neurons for each layer, excluding the input layer.
            // m_a1 is equal to the input values.
            std::vector<float> m_a1;
            std::vector<float> m_a2;
            std::vector<float> m_a3;

            // Maxtrix that will store weights between input layer and first hidden layer
            std::vector<std::vector<float> > m_W1;
            // Maxtrix that will store weights between first hidden layer and second hidden layer.
            std::vector<std::vector<float> > m_W2;
            // Maxtrix that will store weights between second hidden layer and output layer.
            std::vector<std::vector<float> > m_W3;



    };
}

#endif // End of CPU_NEURAL_NETWORK