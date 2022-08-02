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
                                  m_W3(layer_q_size),
                                  m_dLdW1(layer_p_size, 3),
                                  m_dLdW2(layer_q_size, layer_p_size),
                                  m_dLdW3(layer_q_size)
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
    m_a3 = sigmoid(m_z3);
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
 * @param z The output of the output neuron in the last layer.
 * 
 * @return a The activation of the output neuron.
 * 
 */
double cpu::NeuralNetwork::sigmoid(const double &z)
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
 * Make perdictions from the given z-score standardized test data.
 * The perdictions are made as follows.
 *      1. Iterate through the dataset.
 *      2. For each sample in the test set, perform forward propegation.
 *      3. Take the output m_a3 of the sigmoid activivation from the last layer
 *         can convert the value to 0 or 1 based of the given threeshold
 *         which will the perdiction for that given sample.
 *      4. Store the prediction in y_pred array. 
 * 
 * @param X_test_stand z-score standardized test data.
 * @param threeshold If m_a2 > threeshold, then the prediction is 1.
 *                   Otherwise, the prediction is 0.
 * 
 * @return A vector with predictions for each corresponding sample of
 *         X_test_stand.
 * 
 */
std::vector<double> cpu::NeuralNetwork::perdict( Matrix &X_test_stand, const double& threeshold){
    
    std::vector<double> y_pred(X_test_stand.get_num_rows());


    for (int j=0; j < X_test_stand.get_num_rows(); j++){
        this->m_x = X_test_stand.getRow(j);

        this->forward_propegation();

        if(m_a3 > threeshold){
            y_pred[j] = 1.0;
        }else{
            y_pred[j] = 0.0;
        }
    }

    return y_pred;
}

/**
 * 
 * Compute the accuracy of the model using the following formula
 * @f$accuracy = \frac{correct predictions}{ total number of perdictions}$
 * 
 * @param y_pred The vector holding the perdictions of the model for every test sample
 * @param y_test The vector holding the true outcome for every test sample.
 * 
 * Assumption: y_pred and y_test are of the same size.
 * 
 * @return The accuracy score.
 * 
 */
double cpu::NeuralNetwork::computeAccuracy(std::vector<double>& y_pred, std::vector<double>& y_test){
    double accuracy = 0.0;

    for (int j=0; j < y_test.size(); j++){
        if(y_pred[j] == y_test[j]){
            accuracy++;
        }
    }

    accuracy = accuracy/static_cast<double>(y_test.size());

    return accuracy;
}

/**
 * 
 * Compute the loss of the neural network using the 
 * Cross-Entropy loss function.
 * 
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 * 
 * @param y The actual outcomes.
 * @param a The output of the sigmoid activation neuron.
 * 
 * @return entropy loss 
 * 
 * Assumptions: The values of activations are greater than zero since they are 
 * the result of sigmoid activation.
 * 
 */
double cpu::NeuralNetwork::bceLoss(const double &y, 
                                       const double &a){
    double loss = 0.0f;
    // Use epsilon since log of zero is undefined.
    double epsilon = 0.0001; 


    loss += -y*std::log(a + epsilon) - (1-y)*std::log(1-a + epsilon);



    return loss;
}

/**
 * Compute the derivative of binary cross entropy loss function
 * @f$\frac{\partial L}{\partial a} = - \frac{y}{a} + \fra{1-y}{1-a}$ where
 * @f$a$ is the output of the output neuron.
 * Since division by zero can cause numerical issues, a small value, called epsilon,
 * will be added to the denominator terms.
 * 
 * @param y The outcomes.
 * @param a The output of the sigmoid activation neuron.
 * 
 * @return The derivative of the cross entropy loss function with
 *         respect to the signmoid activation a, in other words, f'(z)
 */
double cpu::NeuralNetwork::bceLossPrime(const double &y, const double &a){

    double loss = 0.0;
    double epsilon = 0.0001;

    loss += -(y/(a+epsilon)) + ((1-y)/(1-a+epsilon));

    return loss;
}

/**
 * Compute the derivative of the sigmoid activiation which can be defined as
 * @f$\sigma^{'} = \sigma(1-\sigma)$.
 * 
 * @param z The output of the output neuron in the last layer.
 * 
 * @return a The sigmoid prime activation.
 */
double cpu::NeuralNetwork::sigmoidPrime(const double& z){

    double a = sigmoid(z)*(1.0 - sigmoid(z));

    return a;
}

/**
 * Compute the error term associated with the output neuron j in the last layer.
 * The error term is commonly referred to as delta and is defined as the following
 * @f$\delta_j = f'(z)\frac{\partial L}{\partial a} = $
 * @f$         \sigma^{'}(z) (- \frac{y}{a} + \fra{1-y}{1-a})$
 * 
 * This function serves as a helper function for computeGradientInit.
 * 
 * @param y The outcomes from the dataset
 * @param a The activation of the sigmoid neuron
 * @param z the output of the sigmoid neuron TODO check if this correct.
 * 
 * @return The error term associated with the output neuron.
 */
double cpu::NeuralNetwork::computeDeltaInit(const double& y,
                                            const double& a,
                                            const double& z){
    double delta = sigmoidPrime(z) * bceLossPrime(y, a);

    return delta;
}

/**
 * 
 * Compute the gradient of the weight between the second hidden layer
 * and the output layer.
 * @f$dL/dW = \delta a$
 * 
 * @param delta The delta term from the output neuron.
 * @param a     A vector containing the outputs of the second hidden layer.
 * 
 * @return dL/dW 
 * 
 */
std::vector<double> cpu::NeuralNetwork::computeGradientInit(const double& delta,
                                        const std::vector<double>& a){
    std::vector<double> dW(a.size()); 

    for(int i =0; i < a.size(); i++){
        dW[i] = delta*a[i];
    }

    return dW;
}


cpu::Matrix& cpu::NeuralNetwork::W1(){
    return m_W1;
}


cpu::Matrix& cpu::NeuralNetwork::W2(){
    return m_W2;
}


std::vector<double>& cpu::NeuralNetwork::W3(){
    return m_W3;
}

void cpu::NeuralNetwork::W1(const Matrix& _W1){
    this->m_W1 = _W1;
}

void cpu::NeuralNetwork::W2(const Matrix& _W2){
    this->m_W2 = _W2;
}

void cpu::NeuralNetwork::W3(const std::vector<double>& _W3){
    this->m_W3 = _W3;
}

const cpu::Matrix& cpu::NeuralNetwork::W1() const{
    return m_W1;
}


const cpu::Matrix& cpu::NeuralNetwork::W2() const{
    return m_W2;
}


const std::vector<double>& cpu::NeuralNetwork::W3() const{
    return m_W3;
}