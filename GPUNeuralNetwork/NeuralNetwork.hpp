#include <memory>
#include "Dataset.hpp"

namespace gpu{
    /**
     * An artifical neural network that will use forward
     * and backpropegation to determine weights that will be used 
     * for perdiction. Gradient decent will be used to compute gradients.
     * 
     */
    class NeuralNetwork{

        public:
            NeuralNetwork(int input_size, int layer1_size, int layer2_size, 
                                  int epoch, int alpha, int output_size);

            void allocatedMemeory();
            void initializeWeights(std::shared_ptr<float>& W, int layer_I, int layer_J);
            void fit(Dataset& X_train_stand);
            
            void forwardPropegation();
            void computeOutputs(std::shared_ptr<float>& z, 
                                const std::shared_ptr<float>& W, 
                                const std::shared_ptr<float>& a,
                                int layerI_size,
                                int layerJ_size);
            void computeOutputs(double& z, 
                                const std::shared_ptr<float>& W, 
                                const std::shared_ptr<float>& a,
                                int layerI_size,
                                int layerJ_size);
            void reluActivation(std::shared_ptr<float>& a,
                                 const std::shared_ptr<float>& z,
                                int layerI_size); 
            double sigmoid(const double& z);

            void x(const std::vector<float>& x);

        private:

            //Store size of each layer
            int m_input_size; // Number of neurons in input layer
            int m_layer1_size; // Number of neurons in first hidden layer
            int m_layer2_size; // Number of neurons in the second hidden layer
            int m_output_size; // Number of neurons in the output layer


            // Store the output of the neurons for each layer, excluding the input layer.
            std::shared_ptr<float> m_z1;
            std::shared_ptr<float> m_z2;
            double m_z3;

            // Store the activations of the neurons for each layer.
            std::shared_ptr<float> m_a1;
            std::shared_ptr<float> m_a2;
            double m_a3;

            // Store the sample input the neural network x
            // and the expected outcome y.
            std::shared_ptr<float> m_x;
            double m_y;

            // Store the error terms for each layer
            std::shared_ptr<float> m_delta1;
            std::shared_ptr<float> m_delta2;
            double m_delta3; 

            // Store weight s between layers.
            // Note W1 and W2 are flatten matrices.
            std::shared_ptr<float> m_W1;
            std::shared_ptr<float> m_W2;
            std::shared_ptr<float> m_W3;

            // Store the number of iterations of training the neural network.
            int m_epoch;
            // Store the step size for gradient descent
            double m_alpha;

            // Store the gradient of the weights for each layer 
            //Note dLdW1 and dLdW2 are flatten matrices.
            std::shared_ptr<float> m_dLdW1;
            std::shared_ptr<float> m_dLdW2;
            std::shared_ptr<float> m_dLdW3;

    };
}