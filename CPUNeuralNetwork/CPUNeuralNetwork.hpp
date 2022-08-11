#ifndef CPU_NEURAL_NETWORK
#define CPU_NEURAL_NETWORK

#include <vector>
#include "Dataset.hpp"
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
                          int layer_q_size,
                          int epoch,
                          double alpha);

            void weight_initialization( Matrix& W);
            void weight_initialization( std::vector<double>& W);

            void fit(Dataset& X_train_stand, std::vector<double>& y_train);

            void forward_propegation();
            
            std::vector<double> compute_outputs(const cpu::Matrix &W, 
                                               const std::vector<double> &a);
            double computeOutputLastLayer(const std::vector<double> &W, 
                                          const std::vector<double> &a);

            std::vector<double> relu_activation(const std::vector<double> &z);

            double sigmoid(const double& z);

            std::vector<double> perdict(Dataset& X_test_stand, const double& threeshold);

            double computeAccuracy(std::vector<double>& y_pred, std::vector<double>& y_test);

            double bceLoss(const double &y, 
                            const double &a);

            void backPropegation();

            double sigmoidPrime(const double& z);

            std::vector<double> reluPrime(const std::vector<double> &z);
                               
            double bceLossPrime(const double &y, 
                                const double &a);

            std::vector<double> computeDeltaInit(const double& y,
                                                const double& a,
                                                const double& z);

            std::vector<double> computeDeltaInit(const std::vector<double>& W,
                                                const std::vector<double>& delta,
                                                const std::vector<double>& z);

            std::vector<double> computeGradientInit(const std::vector<double>& delta,
                                                    const std::vector<double>& a);

            std::vector<double> computeDelta(const Matrix& W, 
                                             const std::vector<double>& delta_,
                                             const std::vector<double>& z);

            cpu::Matrix computeGradient(const std::vector<double>& delta,
                                        const std::vector<double>& a);

            cpu::Matrix gradientDecent(const Matrix& W,
                                        const double& alpha, 
                                        const Matrix& dLdW);
            std::vector<double> gradientDecentInit(const std::vector<double>& W,
                                                    const double& alpha,
                                                    const std::vector<double>& dLdW);
            
            void updateWeigths();


            // Setter and getter methodes

            void x(const std::vector<double>& _x);
            void W1(const cpu::Matrix& _W1);
            void W2(const cpu::Matrix& _W2);
            void W3(const std::vector<double>& _W3);
            void y(const double& _y);

            double& a3();

            const std::vector<double>& dLdW3() const;
            const cpu::Matrix& dLdW2() const;
            const cpu::Matrix& dLdW1() const;

        private:

            // Store the output of the neurons for each layer, excluding the input layer.
            std::vector<double> m_z1;
            std::vector<double> m_z2;
            // The last layer has only one neuron so an std::vector is unnecessary.
            double m_z3;

            // Store the activations of the neurons for each layer.
            std::vector<double> m_a1;
            std::vector<double> m_a2;
            // The last layer has only one neuron so an std::vector is unnecessary.
            double m_a3;

            // Store the sample input the neural network x
            // and the expected outcome y.
            std::vector<double> m_x;
            double m_y;

            // Store the error terms for each layer
            std::vector<double> m_delta1;
            std::vector<double> m_delta2;
            std::vector<double> m_delta3; 

            // Maxtrix that will store weights between input layer and first hidden layer
            Matrix m_W1;
            // Maxtrix that will store weights between first hidden layer and second hidden layer.
            Matrix m_W2;
            // Vector that will store weights between second hidden layer and output layer.
            std::vector<double> m_W3;

            // Store the number of iterations of training the neural network.
            int m_epoch;
            // Store the step size for gradient descent
            double m_alpha;

            // Store the gradient of the weights for each layer 
            Matrix m_dLdW1;
            Matrix m_dLdW2;
            std::vector<double> m_dLdW3;


    };
}

#endif // End of CPU_NEURAL_NETWORK