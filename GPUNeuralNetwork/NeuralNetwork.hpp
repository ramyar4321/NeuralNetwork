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
                                  int output_size, int epoch, int alpha);

            void allocatedMemeory();
            void initializeWeights(std::shared_ptr<float>& W, int layerI_size, int layerJ_size);
            void fit(Dataset& X_train_stand);
            
            void forwardPropegation();
            void computeOutputs(std::shared_ptr<float>& z, 
                                const std::shared_ptr<float>& W, 
                                const std::shared_ptr<float>& a,
                                const int layerI_size,
                                const int layerJ_size);
            void computeOutputs(double& z, 
                                const std::shared_ptr<float>& W, 
                                const std::shared_ptr<float>& a,
                                const int layerI_size,
                                const int layerJ_size);
            void reluActivation(std::shared_ptr<float>& a,
                                const std::shared_ptr<float>& z,
                                const int layerI_size); 
            double sigmoid(const double& z);

            std::vector<float> perdict(Dataset& X_test_stand, const float& threeshold);
            float computeAccuracy(std::vector<float>& y_pred, std::vector<float>& y_test);

            void backPropegation();
            double sigmoidPrime(const double& z);
            void reluPrime(std::shared_ptr<float>& fprime,
                            const std::shared_ptr<float> &z,
                            const int layerI_size);
            double bceLossPrime(const double &y, 
                                const double &a);
            void computeDeltaInit(double& delta,
                                    const double& y,
                                    const double& a,
                                    const double& z);
            void computeDelta(std::shared_ptr<float>& delta,
                                const std::shared_ptr<float>& W, 
                                const std::shared_ptr<float>& delta_,
                                const std::shared_ptr<float>& fprime,
                                const int layerJ_size,
                                const int layerK_size);
            void computeDelta(std::shared_ptr<float>& delta,
                                const std::shared_ptr<float>& W, 
                                const double& delta_,
                                const std::shared_ptr<float>& fprime,
                                const int layerJ_size,
                                const int layerK_size);
            void computeGradient(std::shared_ptr<float>& dLdW,
                                 const std::shared_ptr<float>& delta,
                                 const std::shared_ptr<float>& a,
                                 const int layerI_size,
                                 const int layerJ_size);
            void computeGradientInit(std::shared_ptr<float>& dLdW,
                                     const double& delta,
                                     const std::shared_ptr<float>& a,
                                     const int layerI_size,
                                     const int layerJ_size);

            void updateWeigths();
            void gradientDecent(std::shared_ptr<float>& W,
                                const std::shared_ptr<float>& dLdW,
                                const int layerI_size,
                                const int layerJ_size);

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

            // Store the results of derivative of the ReLu activation function
            std::shared_ptr<float> m_fprime1;
            std::shared_ptr<float> m_fprime2;


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