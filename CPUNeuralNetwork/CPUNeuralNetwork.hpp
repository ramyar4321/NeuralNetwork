#ifndef CPU_NEURAL_NETWORK
#define CPU_NEURAL_NETWORK

#include <vector>
#include "Dataset.hpp"
#include "Matrix.hpp"
#include "Vector.hpp"


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


            void fit(Dataset& X_train_stand, std::vector<double>& y_train);

            void forward_propegation();
            
            cpu::Vector compute_outputs(const cpu::Matrix &W, 
                                        const cpu::Vector &a);
            double compute_outputs(const cpu::Vector &W, 
                                    const cpu::Vector &a);

            cpu::Vector relu_activation(const cpu::Vector &z);

            double sigmoid(const double& z);

            std::vector<double> perdict(Dataset& X_test_stand, const double& threeshold);

            double computeAccuracy(std::vector<double>& y_pred, std::vector<double>& y_test);

            double bceLoss(const double &y, 
                            const double &a);

            void backPropegation();

            double sigmoidPrime(const double& z);

            cpu::Vector reluPrime(const cpu::Vector &z);
                               
            double bceLossPrime(const double &y, 
                                const double &a);

            double computeDeltaInit(const double& y,
                                                const double& a,
                                                const double& z);
            cpu::Vector computeDelta(const Matrix& W, 
                                             const cpu::Vector& delta_,
                                             const cpu::Vector& z);
            cpu::Vector computeDelta(const Vector& W, 
                                    const double& delta_,
                                    const cpu::Vector& z);

            cpu::Matrix computeGradient(const cpu::Vector& delta,
                                        const cpu::Vector& a);
            cpu::Vector computeGradientInit(const double& delta,
                                            const cpu::Vector& a);

            cpu::Matrix gradientDecent(const Matrix& W,
                                        const double& alpha, 
                                        const Matrix& dLdW);
            cpu::Vector gradientDecent(const cpu::Vector& W,
                                        const double& alpha,
                                        const cpu::Vector& dLdW);
            
            void updateWeigths();


            // Setter and getter methodes

            void x(const cpu::Vector& _x);
            void W1(const cpu::Matrix& _W1);
            void W2(const cpu::Matrix& _W2);
            void W3(const cpu::Vector& _W3);
            void y(const double& _y);

            double& a3();

            const cpu::Vector& dLdW3() const;
            const cpu::Matrix& dLdW2() const;
            const cpu::Matrix& dLdW1() const;

        private:

            // Store the output of the neurons for each layer, excluding the input layer.
            cpu::Vector m_z1;
            cpu::Vector m_z2;
            double m_z3;

            // Store the activations of the neurons for each layer.
            cpu::Vector m_a1;
            cpu::Vector m_a2;
            double m_a3;

            // Store the sample input the neural network x
            // and the expected outcome y.
            cpu::Vector m_x;
            double m_y;

            // Store the error terms for each layer
            cpu::Vector m_delta1;
            cpu::Vector m_delta2;
            double m_delta3; 

            // Maxtrix that will store weights between input layer and first hidden layer
            Matrix m_W1;
            Matrix m_W2;
            Vector m_W3;

            // Store the number of iterations of training the neural network.
            int m_epoch;
            // Store the step size for gradient descent
            double m_alpha;

            // Store the gradient of the weights for each layer 
            Matrix m_dLdW1;
            Matrix m_dLdW2;
            Vector m_dLdW3;


    };
}

#endif // End of CPU_NEURAL_NETWORK