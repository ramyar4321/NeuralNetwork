#ifndef CPU_NEURAL_NETWORK
#define CPU_NEURAL_NETWORK

#include <vector>
#include "Dataset.hpp"
#include "Matrices/Matrix.hpp"
#include "Matrices/Vector.hpp"
#include "Layers/HiddenLayer.hpp"
#include "Layers/OutputLayer.hpp"

namespace cpu {
    /** 
     * Class NeuralNetwork
     * 
     * The neural network will be comprised of an input layer, two hidden layers,
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
                          float alpha);


            void fit(Dataset& X_train_stand, std::vector<float>& y_train);
            std::vector<float> perdict(Dataset& X_test_stand, const float& threeshold);
            float computeAccuracy(std::vector<float>& y_pred, std::vector<float>& y_test);


            float forwardPropegation();
            void backPropegation();      
            void updateWeigths();


            cpu::HiddenLayer m_hidden_layer1;
            cpu::HiddenLayer m_hidden_layer2;
            cpu::OutputLayer m_output_layer;

            // Setter methods
            void x(const cpu::Vector& x);
            void y(const float& y);



        private:

            // Store the number of iterations of training the neural network.
            int m_epoch;
            // Store the step size for gradient descent
            float m_alpha;

            // Use variable to store the sample from the dataset 
            // to be passed to forward and back propegation methodes.
            cpu::Vector m_x;
            // Use variable to store the outcome associated
            // with each given sample from the dataset to be 
            // passed to backward propegation methode.
            float m_y;


    };
}

#endif // End of CPU_NEURAL_NETWORK