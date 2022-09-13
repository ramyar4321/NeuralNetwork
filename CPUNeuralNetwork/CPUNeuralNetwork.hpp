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
            NeuralNetwork(int epoch,
                          double alpha);

            void addLayer(cpu::Layer* layer);


            void fit(Dataset& X_train_stand, std::vector<double>& y_train);
            std::vector<double> perdict(Dataset& X_test_stand, const double& threeshold);
            double computeAccuracy(std::vector<double>& y_pred, std::vector<double>& y_test);

            void weightInitialization();

            cpu::Vector forwardPropegation();
            void backPropegation();      
            void updateWeigths();

            void x(const cpu::Vector& x);
            void W(const cpu::Matrix& W_, const int& layer_index);

        private:

            // Use variable to store the sample from the dataset 
            // to be passed to forward and back propegation methodes.
            cpu::Vector m_x;
            // Use variable to store the outcome associated
            // with each given sample from the dataset to be 
            // passed to backward propegation methode.
            cpu::Vector m_y;

            // Vector to store the layers of Neural Network.
            std::vector<cpu::Layer*> m_layers;

            // Number of layers in neural network;
            int m_num_layers;

            // Store the number of iterations of training the neural network.
            int m_epoch;
            // Store the step size for gradient descent
            double m_alpha;


    };
}

#endif // End of CPU_NEURAL_NETWORK