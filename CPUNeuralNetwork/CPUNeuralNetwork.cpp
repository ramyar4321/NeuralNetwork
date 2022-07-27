#include "CPUNeuralNetwork.hpp"
#include <iostream>
#include "random"

/**
 * Initialize Neural Network memeber variables.
 */
cpu::NeuralNetwork::NeuralNetwork(int layer_p_size,
                                  int layer_q_size):
                                  m_z1(layer_p_size, 0.0),
                                  m_z2(layer_q_size, 0.0),
                                  m_z3(0.0),
                                  m_a1(layer_p_size, 0.0),
                                  m_a2(layer_q_size, 0.0),
                                  m_a3(0.0),
                                  // Initialize weights of the neural network to be zeros.
                                  // Later on, the weights will be re-initialized using a more 
                                  // sophicticated methode.
                                  m_W1(layer_p_size, 3),
                                  m_W2(layer_q_size, layer_p_size),
                                  m_W3(layer_q_size)
{}

/**
 * Use data to train the neural network.
 * 
 * @param X_train_stand The X train dataset used to train the Neural Network.
 *                      The X train dataset is assumed to be z-score standardized.
 * @param y_train       The y train dataset used to train the Neural Network.
 *                      The y train dataset is assumed to have values of 0 or 1.
 */
void cpu::NeuralNetwork::fit(Matrix &X_train_stand, std::vector<double>& y_train){

    weight_initialization(m_W1);
    weight_initialization(m_W2);
    weight_initialization(m_W3);

    for (int j=0; j < X_train_stand.get_num_rows(); j++){
        this->m_x = X_train_stand.getRow(j);

        this->forward_propegation();
    }
}

/**
 * Perform forward propegation.
 * 
 * TODO currently implementation 
 * has unnessesary copying of vectors.
 */
void cpu::NeuralNetwork::forward_propegation(){
    m_z1 = compute_outputs(m_W1, m_x);
    m_a1 = relu_activation(m_z1);

    m_z2 = compute_outputs(m_W2, m_a1);
    m_a2 = relu_activation(m_z2);

    m_z3 = computeOutputLastLayer(m_W3, m_a2);
    m_a3 = sigmoid_activation(m_z3);
}

/**
 * Initialize the weigths of the layer to random values that come
 * from a Gaussian Distribtuion centered at 0 with standard deviations of 
 * @f$\sqrt{ \farc{1}{n_{I}}} $ where @f$n_{I}$ is the size of layer @f$I$.
 * 
 * @param W The matrix that contains the weigths connecting 
 *          the neurons of layer I to the neurons of layer J.
 * 
 */
void cpu::NeuralNetwork::weight_initialization(cpu::Matrix& W)
{


    std::mt19937 generator;
    double mean = 0.0f;
    double stddev = std::sqrt(1 / static_cast<double>(W.get_num_cols()) ); 
    std::normal_distribution<double> normal(mean, stddev);
    for (unsigned int j=0; j< W.get_num_rows(); ++j) {
        for (unsigned int i=0; i< W.get_num_cols(); ++i) {
            W[j][i] = normal(generator);
        }
    } 

}

/**
 * Initialize the weigths of the layer to random values that come
 * from a Gaussian Distribtuion centered at 0 with standard deviations of 
 * @f$\sqrt{ \farc{1}{n_{I}}} $ where @f$n_{I}$ is the size of layer @f$I$.
 * 
 * @param W The vector that contains the weigths connecting 
 *          the neurons of layer I to the neuron of the last layer.
 * 
 */
void cpu::NeuralNetwork::weight_initialization(std::vector<double>& W)
{


    std::mt19937 generator;
    double mean = 0.0f;
    double stddev = std::sqrt(1 / static_cast<double>(W.size()) ); 
    std::normal_distribution<double> normal(mean, stddev);
    for (unsigned int i=0; i< W.size(); ++i) {
        W[i] = normal(generator);
    }


}

/**
 * Compute the output of each neuron j in layer J. 
 * The output for each neuron can be computed as follows 
 * @f$z_j = \sum_{i}^I w_{ji} a_i$ where @f$a_i$ is the output of neuron i
 * from the pervious layer I.
 * 
 * @param W The matrix that contains the weigths connecting the neurons of layer I 
 *          to the neurons of layer J.
 * @param a The vector that contains the activations of each neuron in layer I
 * 
 * @return z The vector that contains the output of each neuron in layer J
 * 
 */
std::vector<double> cpu::NeuralNetwork::compute_outputs(const cpu::Matrix &W, 
                                                       const std::vector<double> &a)
{
    std::vector<double> z(W.get_num_rows(), 0.0f);

    for (unsigned int j=0; j< W.get_num_rows(); j++) {
        for (unsigned int i=0; i< W.get_num_cols(); i++) {
            z[j] += W[j][i] * a[i];
        }
    } 

    return z;
}

/**
 * This function computes the output of the neuron
 * in the last layer of the neural network. 
 * The output for such a neuron can be computed as follows 
 * @f$z = \sum_{i}^I w_{i} a_i$ where @f$a_i$ is the output of neuron i
 * from the pervious layer I.
 */
double cpu::NeuralNetwork::computeOutputLastLayer(const std::vector<double> &W, 
                                           const std::vector<double> &a){

    double z;

    for (unsigned int i=0; i< W.size(); i++) {
        z += W[i] * a[i];
    }

    return z;

}

/**
 * Compute the activation of each neuron j in layer J using the ReLu activation function. 
 * The activation for each neuron can be computed as follows 
 * @f$z_j = max(0, z_j)$. This method should be called for
 * hidden layers of the neural network.
 * 
 * @param z The vector that contains the output of each neuron in layer J
 * 
 * @return a The vector that contains the activations of each neuron in layer J
 * 
 */
std::vector<double> cpu::NeuralNetwork::relu_activation(const std::vector<double> &z)

{
    std::vector<double> a(z.size(), 0.0f); 

    for (unsigned int j=0; j<z.size(); j++) {
        if(z[j] > 0.0f ){
            a[j] = z[j];
        }else{
            a[j] = 0.0f;
        }
    } 

    return a;
}

/**
 * Compute the sigmoid activation of the output neuron.
 * 
 * The sigmoid activation function for the output neuron can be defined as the following
 * @f$\sigma (z_3) = \frac{1}{1+ \exp (- z_3)} = \frac{\exp (z_3)}{1+ \exp ( z_3)}$. 
 * If z_3 is positive and its magnitude large, it can cause overflow when computing @f$\exp ( z_3)$ and
 * if z_3 is negative and its magnitude is too large, it can cause overflow when computing @f$\exp (- z_3)$.
 * In order to avoid numerical instability, let @f$\sigma (z_3) = \frac{1}{1+ \exp (- z_3)}$ for @f$ z_3 >=0 $
 * and let @f$\sigma (z_3) = \frac{\exp (z_3)}{1+ \exp ( z_3)}$ for @f$ z_3 < 0 $.
 * 
 * @see https://stackoverflow.com/questions/41800604/need-help-understanding-the-caffe-code-for-sigmoidcrossentropylosslayer-for-mult
 * @see https://stackoverflow.com/questions/40353672/caffe-sigmoidcrossentropyloss-layer-loss-function
 * 
 * @param z The output of the output nueron in the last layer.
 * 
 * @return a The activation of the output neuron.
 * 
 */
double cpu::NeuralNetwork::sigmoid_activation(const double &z)
{
    double a = 0.0; 
    if (z >= 0.0f) {
        a = 1.0f / (1.0f + std::exp(-z));
    } else {
        a = std::exp(z) / (1.0f + std::exp(z));
    }

    return a;
}

/**
 * 
 * Compute the loss of the neural network using the 
 * Corss-Entropy loss function.
 * 
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 * 
 * @param y The vector that contains the actual outcomes.
 * @param a The vector that contains the activations of each neuron in layer J
 *          where J is the output layer. In this case, a is the perdicted 
 *          outcome of the neural network.
 * 
 * @return entropy loss 
 * 
 * Assumptions: The values of activations are greater than zero since they are 
 * the result of sigmoid activation.
 * 
 */
double cpu::NeuralNetwork::compute_loss(const double &y, 
                                       const double &a){
    double loss = 0.0f;
    // Use epsilon since log of zero is undefined.
    double epsilon = 0.0001; 


    loss += -y*std::log(a + epsilon) - (1-y)*std::log(1-a + epsilon);



    return loss;
}