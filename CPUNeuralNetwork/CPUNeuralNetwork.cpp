#include "CPUNeuralNetwork.hpp"
#include <iostream>
#include "random"

/**
 * Intialize Neural Network memeber variables.
 */
cpu::NeuralNetwork::NeuralNetwork(unsigned int input_size,
                                  unsigned int layer_p_size,
                                  unsigned int layer_q_size,
                                  unsigned int layer_r_size):
                                  m_input_size(input_size),
                                  m_layer_p_size(layer_p_size),
                                  m_layer_q_size(layer_q_size),
                                  m_layer_r_size(layer_r_size),
                                  m_z1(layer_p_size, 1),
                                  m_z2(layer_q_size, 1),
                                  m_z3(layer_r_size, 1),
                                  m_a1(layer_p_size, 1),
                                  m_a2(layer_q_size, 1),
                                  m_a3(layer_r_size, 1),
                                  // Initialize weights of the neural network to be ones.
                                  // Later on, the weights will be re-initialized using a more 
                                  // sophicticated methode.
                                  m_W1(layer_p_size, std::vector<float>(input_size, 1)),
                                  m_W2(layer_q_size, std::vector<float>(layer_p_size,1)),
                                  m_W3(layer_r_size, std::vector<float>(layer_q_size, 1))
{}

/**
 * Use data to train the neural network.
 */
void cpu::NeuralNetwork::fit(){

    m_W1 = weight_initialization(m_input_size, m_layer_p_size);
    m_W2 = weight_initialization(m_layer_p_size, m_layer_q_size );
    m_W3 = weight_initialization(m_layer_q_size, m_layer_r_size);

    //forward_propegation();
}

/**
 * Initialize the weigths of the neural network to random values that come
 * from a Gaussian Distribtuion centered at 0 with standard deviations of 
 * @f$\sqrt{ \farc{1}{n_{I}}} $ where @f$n_{I}$ is the size of layer @f$I$.
 * 
 * @param layer_i_size The number of neurons in layer I.
 * @param layer_j_size The number of neurons in layer J.
 * 
 * @return W The matrix that contains the weigths connecting the neurons of layer I to the neurons of layer J.
 * 
 */
std::vector<std::vector<float> > cpu::NeuralNetwork::weight_initialization( const unsigned int &layer_i_size, 
                                                                            const unsigned int &layer_j_size)
{

    std::vector<std::vector<float> > W(layer_j_size, std::vector<float>(layer_i_size, 1.1f));

    std::mt19937 generator;
    float mean = 0.0f;
    float stddev = std::sqrt(1 / static_cast<float>(layer_i_size) ); 
    std::normal_distribution<double> normal(mean, stddev);
    for (unsigned int j=0; j<layer_j_size; ++j) {
        for (unsigned int i=0; i<layer_i_size; ++i) {
            W[j][i] = normal(generator);
        }
    } 

    return W;
}

/**
 * Compute the output of each neuron j in layer J. 
 * The output for each neuron can be computed as follows 
 * @f$z_j = \sum_{i}^I w_{ji} a_i$ where @f$a_i$ is the output of neuron i
 * from the pervious layer I.
 * 
 * @param W The matrix that contains the weigths connecting the neurons of layer I to the neurons of layer J.
 * @param a The vector that contains the activations of each neuron in layer I
 * @param layer_i_size The number of neurons in layer I.
 * @param layer_j_size The number of neurons in layer J.
 * 
 * @return z The vector that contains the output of each neuron in layer J
 * 
 */
std::vector<float> cpu::NeuralNetwork::compute_outputs(const std::vector<std::vector<float> > &W, 
                                   const std::vector<float> &a,  
                                   const unsigned int &layer_i_size, 
                                   const unsigned int &layer_j_size)
{
    std::vector<float> z(layer_j_size, 0.0f);

    for (unsigned int j=0; j<layer_j_size; j++) {
        for (unsigned int i=0; i<layer_i_size; i++) {
            z[j] += W[j][i] * a[i];
        }
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
 * @param layer_j_size The number of neurons in layer J.
 * 
 * @return a The vector that contains the activations of each neuron in layer J
 * 
 */
std::vector<float> cpu::NeuralNetwork::relu_activation(const std::vector<float> &z,
                                   const unsigned int &layer_j_size)

{
    std::vector<float> a(layer_j_size, 0.0f); 

    for (unsigned int j=0; j<layer_j_size; j++) {
        if(z[j] > 0.0f ){
            a[j] = z[j];
        }else{
            a[j] = 0.0f;
        }
    } 

    return a;
}

/**
 * Compute the activation of each neuron j in layer J using the sigmoid activation function. 
 * 
 * The sigmoid activation function for a given neuron j of layer J can be defined as the following
 * @f$\sigma (z_j) = \frac{1}{1+ \exp (- z_j)} = \frac{\exp (z_j)}{1+ \exp ( z_j)}$. 
 * If z_j is positive and its magnitude large, it can cause overflow when computing @f$\exp ( z_j)$ and
 * if z_j is negative and its magnitude is too large, it can cause overflow when computing @f$\exp (- z_j)$.
 * In order to avoid numerical instability, let @f$\sigma (z_j) = \frac{1}{1+ \exp (- z_j)}$ for @f$ z_j >=0 $
 * and let @f$\sigma (z_j) = \frac{\exp (z_j)}{1+ \exp ( z_j)}$ for @f$ z_j < 0 $.
 * 
 * @see https://stackoverflow.com/questions/41800604/need-help-understanding-the-caffe-code-for-sigmoidcrossentropylosslayer-for-mult
 * @see https://stackoverflow.com/questions/40353672/caffe-sigmoidcrossentropyloss-layer-loss-function
 * 
 * @param z The vector that contains the output of each neuron in layer J
 *          where J is the output layer
 * @param layer_j_size The number of neurons in layer J.
 * 
 * @return a The vector that contains the activations of each neuron in layer J
 *          where J is the output layer
 * 
 */
std::vector<float> cpu::NeuralNetwork::sigmoid_activation(const std::vector<float> &z,
                                      const unsigned int &layer_j_size)
{
    std::vector<float> a(layer_j_size, 0.0f); 
    for (unsigned int j=0; j<layer_j_size; j++) {
        if (z[j] >= 0.0f) {
            a[j] = 1.0f / (1.0f + std::exp(-z[j]));
        } else {
            a[j] = std::exp(z[j]) / (1.0f + std::exp(z[j]));
        }
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
 * @param layer_j_size The number of neurons in layer J.
 * 
 */
float cpu::NeuralNetwork::compute_loss(const std::vector<float> &y, 
                                       const std::vector<float> &a,
                                       const unsigned int &layer_j_size){
    float loss = 0.0f;

    for (unsigned int j=0; j<layer_j_size; j++) {
        loss += y[j]*std::log(a[j]) + (1-y[j])*std::log(1-a[j]);
    }

    return loss;
}