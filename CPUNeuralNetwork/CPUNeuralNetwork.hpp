#ifndef CPU_NEURAL_NETWORK
#define CPU_NEURAL_NETWORK

#include <vector>
#include "Matrix.hpp"


namespace cpu {
    /** 
     * Class NeuralNetwork
     * 
     * The neural network will of an input layer, two hidden layers,
     * and an output layer. Since the Haberman Breast Cancer dataset
     * will be used, the input layer will be of size 3 and the output layer
     * will be of size 1. The sizes of the two hidden layers will be 
     * specified by the user during instatiation of the object.
     * 
     */
    class NeuralNetwork{

        public:
            NeuralNetwork(int layer_p_size,
                          int layer_q_size);

            void weight_initialization( Matrix& W);
            void weight_initialization( std::vector<double>& W);

            
            void fit(Matrix &X_train_stand, std::vector<double>& y_train);

            void forward_propegation();
            
            std::vector<double> compute_outputs(const cpu::Matrix &W, 
                                               const std::vector<double> &a);
            double computeOutputLastLayer(const std::vector<double> &W, 
                                          const std::vector<double> &a);

            std::vector<double> relu_activation(const std::vector<double> &z);

            double sigmoid_activation(const double &z);

            double compute_loss(const double &y, 
                                const double &a);

        private:

            // Store the output of the neurons for each layer, excluding the input layer.
            std::vector<double> m_z1;
            std::vector<double> m_z2;
            // The last layer has only one neuron so an std::vector is unnecessary.
            double m_z3;

            // Store the activations of the neurons for each layer.
            std::vector<double> m_x;
            std::vector<double> m_a1;
            std::vector<double> m_a2;
            // The last layer has only one neuron so an std::vector is unnecessary.
            double m_a3;

            // Maxtrix that will store weights between input layer and first hidden layer
            Matrix m_W1;
            // Maxtrix that will store weights between first hidden layer and second hidden layer.
            Matrix m_W2;
            // Vector that will store weights between second hidden layer and output layer.
            std::vector<double> m_W3;



    };
}

#endif // End of CPU_NEURAL_NETWORK