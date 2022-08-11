#include "CPUNeuralNetwork.hpp"
#include <iostream>
#include "random"

/**
 * Initialize Neural Network memeber variables.
 */
cpu::NeuralNetwork::NeuralNetwork(int layer_p_size,
                                  int layer_q_size,
                                  int epoch,
                                  double alpha):
                                  m_z1(layer_p_size, 0.0),
                                  m_z2(layer_q_size, 0.0),
                                  m_z3(0.0),
                                  m_a1(layer_p_size, 0.0),
                                  m_a2(layer_q_size, 0.0),
                                  m_a3(0.0),
                                  m_x(3, 0.0),
                                  m_y(0.0),
                                  // Initialize weights of the neural network to be zeros.
                                  // Later on, the weights will be re-initialized using a more 
                                  // sophicticated methode.
                                  m_W1(layer_p_size, 3),
                                  m_W2(layer_q_size, layer_p_size),
                                  m_W3(layer_q_size),
                                  m_epoch(epoch),
                                  m_alpha(alpha),
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
void cpu::NeuralNetwork::fit(cpu::Dataset& X_train_stand, std::vector<double>& y_train){

    weight_initialization(m_W1);
    weight_initialization(m_W2);
    weight_initialization(m_W3);

    for (int e =0; e< this->m_epoch; e++){
        for (int j=0; j < X_train_stand.get_num_rows(); j++){
            this->m_x = X_train_stand.getRow(j);

            this->forward_propegation();
            this->backPropegation();
            this->updateWeigths();
        }
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

    z = W*a;

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
std::vector<double> cpu::NeuralNetwork::perdict( cpu::Dataset& X_test_stand, const double& threeshold){
    
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
 * Perform back propegation
 */
void cpu::NeuralNetwork::backPropegation(){
    this->m_delta3 = this->computeDeltaInit(this->m_y, this->m_a3, this->m_z3);
    this->m_dLdW3 =  this->computeGradientInit(this->m_delta3, this->m_a2);

    this->m_delta2 = this->computeDeltaInit(this->m_W3, this->m_delta3, this->m_z2);
    this->m_dLdW2 =  this->computeGradient(this->m_delta2, this->m_a1);

    this->m_delta1 = this->computeDelta(this->m_W2, this->m_delta2, this->m_z1);
    this->m_dLdW1 = this->computeGradient(this->m_delta1, this->m_x);

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
 * Compute activation of neuron i of layer I using the derivative of ReLu
 * activation function.
 * @f$\begin{math}
        f'(z_i)=\left\{
            \begin{array}{ll}
                0, & \mbox{if $x<0$}.\\
                1, & \mbox{if $x>0$}.
            \end{array}
        \right.
    \end{math}$
 * The derivative is undefined at z_i = 0 but it can be set to zero
 * in order to produce sparse vector.
 * 
 * @param z A vector that contains the output of each neuron in layer I
 * 
 * @return A vector containing f' for each neuron in layer I
 */
std::vector<double> cpu::NeuralNetwork::reluPrime(const std::vector<double> &z){
    std::vector<double> f_prime(z.size());

    for (int i = 0; i < z.size(); i++){
        if(z[i] <= 0){
            f_prime[i] = 0;
        }else{
            f_prime[i] = 1;
        }
    }

    return f_prime;
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
std::vector<double> cpu::NeuralNetwork::computeDeltaInit(const double& y,
                                            const double& a,
                                            const double& z){

    std::vector<double> delta = {sigmoidPrime(z) * bceLossPrime(y, a)};

    return delta;
}

/**
 * For layers J < K, compute the error term associated with each neuron j of layer k.
 * @f$\delta_j = f'(z_j)\sum_{k=0}^{n_K} w_{kj} \delta_k$ where
 * @f$f'$ is the derivative of the ReLu activation function,
 * @f$z_j$ is the output of neuron j of layer J, @f$n_K$
 * is the number of neurons in layer K, @f$w_{ji}$ is the
 * weight from neuron j of layer J to neuron k of layer K,
 * and @f$\delta_k$ is the error term of neuron k of layer K.
 * 
 * @param W A matrix containing the weigths between layer I and J
 * @param delta_ A vector cotaining the error terms of each neuron k of layer K
 * @param z a vector containing the output of neuron i of layer I
 * 
 * @return A vector containing the error terms of each neuron i of layer I
 * 
 */
std::vector<double> cpu::NeuralNetwork::computeDeltaInit(const std::vector<double>& W, 
                                 const std::vector<double>& delta_,
                                 const std::vector<double>& z){

    std::vector<double> delta(W.size(), 0.0);

    std::vector<double> f_prime = this->reluPrime(z);


    for(int j=0; j < W.size(); j++){

        delta[j] += W[j] * delta_[0];

        delta[j] *= f_prime[j];
    }


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
std::vector<double> cpu::NeuralNetwork::computeGradientInit(const std::vector<double>& delta,
                                        const std::vector<double>& a){
    std::vector<double> dW(a.size()); 


    for(int i =0; i < a.size(); i++){
        dW[i] = delta[0]*a[i];
    }

    return dW;
}



/**
 * For layers J < K, compute the error term associated with each neuron j of layer k.
 * @f$\delta_j = f'(z_j)\sum_{k=0}^{n_K} w_{kj} \delta_k$ where
 * @f$f'$ is the derivative of the ReLu activation function,
 * @f$z_j$ is the output of neuron j of layer J, @f$n_K$
 * is the number of neurons in layer K, @f$w_{ji}$ is the
 * weight from neuron j of layer J to neuron k of layer K,
 * and @f$\delta_k$ is the error term of neuron k of layer K.
 * 
 * @param W A matrix containing the weigths between layer I and J
 * @param delta_k A vector cotaining the error terms of each neuron j of layer J
 * @param z a vector containing the output of neuron i of layer I
 * 
 * @return A vector containing the error terms of each neuron i of layer I
 * 
 */
std::vector<double> cpu::NeuralNetwork::computeDelta(const cpu::Matrix& W, 
                                 const std::vector<double>& delta_,
                                 const std::vector<double>& z){

    std::vector<double> delta(W.get_num_cols(), 0.0);
    std::vector<double> f_prime = this->reluPrime(z);

    cpu::Matrix W_tranpose = W.transpose();
    delta = W_tranpose*delta_;


    for(int j=0; j < delta.size(); j++){
        delta[j] *= f_prime[j];
    }


    return delta;
}

/**
 * 
 * Compute the gradient for each weight for a 
 * given layer except the last layer of the neural network.
 * For layers I < J, the gradient for any given weight can be computed as follows.
 * @f$\frac{dL}{dw_{ji}} = a_i * \delta_{j}$ where @f$w_{ji}$ is the weight from
 * neuron i of layer I to neuron j of layer J, @f$a_i$ is the activation of neuron
 * i of layer I, and @f$\delta_{j}$ is the error term of neuron j of layer J.
 * 
 * @param detla A vector containing the error terms for each neuron of layer J
 * @param a     A vector cotaining the activation of each neuron of layer I
 * 
 * @return A Matrix containing the weigth between layer I and layer J
 * 
 */
cpu::Matrix cpu::NeuralNetwork::computeGradient(const std::vector<double>& delta,
                            const std::vector<double>& a){
    int num_rows = delta.size();
    int num_cols = a.size();
    cpu::Matrix dW(num_rows,num_cols );

    for(int i =0; i < num_cols; i++){
        for (int j=0; j < num_rows; j++){
            dW[j][i] = a[i]*delta[j];
        }
    }

    return dW;
}

/**
 * 
 * Perform gradient descent. For any given weight between layers I < J,
 * the weight can be updated using the following.
 * @f$ w_{ji} = w_{ji} - \alpha \frac{dL}{dW_{ji}}$
 * 
 * @param W The vector containing the weights between layer I and J
 * @param alpha The step size of gradient descent
 * @param dLdW The vector containing the derivatives of the weights between layer I and J
 * 
 * @return A matrix containing the updated weights between layer I and J
 * 
 */
std::vector<double> cpu::NeuralNetwork::gradientDecentInit(const std::vector<double>& W,
                                                        const double& alpha,
                                                        const std::vector<double>& dLdW){

    std::vector<double> updatedWeights(W.size());

    for(int i=0; i < updatedWeights.size(); i++){
        updatedWeights[i] = updatedWeights[i] - alpha*dLdW[i]; 
    }


    return updatedWeights;

}

/**
 * 
 * Perform gradient descent. For any given weight between layers I < J,
 * the weight can be updated using the following.
 * @f$ w_{ji} = w_{ji} - \alpha \frac{dL}{dW_{ji}}$
 * 
 * @param W The matrix containing the weights between layer I and J
 * @param alpha The step size of gradient descent
 * @param dLdW The maxtrix containing the derivatives of the weights between layer I and J
 * 
 * @return A matrix containing the updated weights between layer I and J
 * 
 */
cpu::Matrix cpu::NeuralNetwork::gradientDecent(const Matrix& W,
                                                const double& alpha,
                                                const Matrix& dLdW){

    cpu::Matrix updatedWeights(W.get_num_rows(), W.get_num_cols());

    updatedWeights = updatedWeights - dLdW*alpha;

    return updatedWeights;

}

/**
 * 
 * Update weights using gradient descent.
 * 
 */
void cpu::NeuralNetwork::updateWeigths(){
    this->m_W3 = this->gradientDecentInit(this->m_W3, this->m_alpha, this->m_dLdW3);
    this->m_W2 = this->gradientDecent(this->m_W2, this->m_alpha, this->m_dLdW2);
    this->m_W1 = this->gradientDecent(this->m_W1, this->m_alpha, this->m_dLdW1);
}

void cpu::NeuralNetwork::x(const std::vector<double>& _x){
    this->m_x = _x;
}


void cpu::NeuralNetwork::W1(const cpu::Matrix& _W1){
    this->m_W1 = _W1;
}


void cpu::NeuralNetwork::W2(const cpu::Matrix& _W2){
    this->m_W2 = _W2;
}


void cpu::NeuralNetwork::W3(const std::vector<double>& _W3){
    this->m_W3 = _W3;
}

void cpu::NeuralNetwork::y(const double& _y){
    this->m_y = _y;
}

double& cpu::NeuralNetwork::a3(){
    return this->m_a3;
}

const std::vector<double>& cpu::NeuralNetwork::dLdW3() const{
    return m_dLdW3;
}

const cpu::Matrix& cpu::NeuralNetwork::dLdW2() const{
    return this->m_dLdW2;
}

const cpu::Matrix& cpu::NeuralNetwork::dLdW1() const{
    return this->m_dLdW1;
}