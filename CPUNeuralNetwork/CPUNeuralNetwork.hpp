#ifndef CPU_NEURAL_NETWORK
#define CPU_NEURAL_NETWORK

#include <vector>
#include "Dataset.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"
#include "Layers/HiddenLayer.hpp"
#include "Layers/OutputLayer.hpp"

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
            NeuralNetwork(int hidden_layer1_size,
                          int hidden_layer2_size,
                          int epoch,
                          double alpha);


            void fit(Dataset& X_train_stand, std::vector<double>& y_train);
            std::vector<double> perdict(Dataset& X_test_stand, const double& threeshold);
            double computeAccuracy(std::vector<double>& y_pred, std::vector<double>& y_test);


            double forwardPropegation();
            void backPropegation();      
            void updateWeigths();


            cpu::HiddenLayer m_hidden_layer1;
            cpu::HiddenLayer m_hidden_layer2;
            cpu::OutputLayer m_output_layer;

            // Setter methods
            void x(const cpu::Vector& x);
            void y(const double& y);



        private:

            // Store the number of iterations of training the neural network.
            int m_epoch;
            // Store the step size for gradient descent
            double m_alpha;

            // Use variable to store the sample from the dataset 
            // to be passed to forward and back propegation methodes.
            cpu::Vector m_x;
            // Use variable to store the outcome associated
            // with each given sample from the dataset to be 
            // passed to backward propegation methode.
            double m_y;


    };
}

#endif // End of CPU_NEURAL_NETWORK