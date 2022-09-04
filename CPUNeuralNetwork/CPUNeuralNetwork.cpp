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
                                  m_z3(0.0f),
                                  m_a1(layer_p_size, 0.0),
                                  m_a2(layer_q_size, 0.0),
                                  m_a3(0.0),
                                  m_x(3, 0.0),
                                  m_y(0.0),
                                  m_delta1(layer_p_size, 0.0f),
                                  m_delta2(layer_q_size, 0.0f),
                                  m_delta3(0.0f),
                                  // Initialize weights of the neural network to be zeros.
                                  // Later on, the weights will be re-initialized using a more 
                                  // sophicticated methode.
                                  m_W1(layer_p_size, 3),
                                  m_W2(layer_q_size, layer_p_size),
                                  m_W3(layer_q_size, 0.0f),
                                  m_epoch(epoch),
                                  m_alpha(alpha),
                                  m_dLdW1(layer_p_size, 3),
                                  m_dLdW2(layer_q_size, layer_p_size),
                                  m_dLdW3(layer_q_size, 0.0f)
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

    m_W1.matrix_initialization();
    m_W2.matrix_initialization();
    m_W3.vectorInitialization();

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
    compute_outputs(m_z1, m_W1, m_x);
    relu_activation(m_a1, m_z1);

    compute_outputs(m_z2, m_W2, m_a1);
    relu_activation(m_a2,m_z2);

    compute_outputs(m_z3, m_W3, m_a2);
    m_a3 = sigmoid(m_z3);
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
void cpu::NeuralNetwork::compute_outputs(cpu::Vector& z, 
                                                const cpu::Matrix &W, 
                                                const cpu::Vector &a)
{

    z = W*a;

}

/**
 * This function computes the output of the neuron
 * in the last layer of the neural network. 
 * The output for such a neuron can be computed as follows 
 * @f$z = \sum_{i}^I w_{i} a_i$ where @f$a_i$ is the output of neuron i
 * from the pervious layer I.
 */
void cpu::NeuralNetwork::compute_outputs(double& z,
                                        const cpu::Vector &W, 
                                        const cpu::Vector &a){


    z = W.dot(a);

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
void cpu::NeuralNetwork::relu_activation(cpu::Vector& a,
                                                        const cpu::Vector &z)

{

    for (int j=0; j<z.getSize(); j++) {
        if(z[j] > 0.0f ){
            a[j] = z[j];
        }else{
            a[j] = 0.0f;
        }
    } 
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
double cpu::NeuralNetwork::sigmoid(const double& z)
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
void cpu::NeuralNetwork::computeDelta(cpu::Vector& delta,
                                            const cpu::Vector& W, 
                                            const double& delta_,
                                            const cpu::Vector& z){

    cpu::Vector f_prime = this->reluPrime(z);

    delta = W*delta_;
    delta *= f_prime;

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
void cpu::NeuralNetwork::computeGradient(cpu::Matrix& dLdW,
                                                const cpu::Vector& delta,
                                                const cpu::Vector& a){
    int num_rows = delta.getSize();
    int num_cols = a.getSize();

    dLdW = a.tensor(delta);
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
void cpu::NeuralNetwork::computeGradientInit(cpu::Vector& dLdW,
                                                    const double& delta,
                                                    const cpu::Vector& a){

    dLdW = a*delta;

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
void cpu::NeuralNetwork::gradientDecent(const Matrix& W,
                                                const double& alpha,
                                                const Matrix& dLdW){

    cpu::Matrix updatedWeights(W.get_num_rows(), W.get_num_cols());

    updatedWeights = updatedWeights - dLdW*alpha;


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
void cpu::NeuralNetwork::gradientDecent(const cpu::Vector& W,
                                                const double& alpha,
                                                const cpu::Vector& dLdW){

    cpu::Vector updatedWeights(W.getSize(), 0.0f);
    updatedWeights -= dLdW*alpha;

}

/**
 * 
 * Update weights using gradient descent.
 * 
 */
void cpu::NeuralNetwork::updateWeigths(){
    this->gradientDecent(this->m_W3, this->m_alpha, this->m_dLdW3);
    this->gradientDecent(this->m_W2, this->m_alpha, this->m_dLdW2);
    this->gradientDecent(this->m_W1, this->m_alpha, this->m_dLdW1);
}

void cpu::NeuralNetwork::x(const cpu::Vector& _x){
    this->m_x = _x;
}


void cpu::NeuralNetwork::W1(const cpu::Matrix& _W1){
    this->m_W1 = _W1;
}


void cpu::NeuralNetwork::W2(const cpu::Matrix& _W2){
    this->m_W2 = _W2;
}


void cpu::NeuralNetwork::W3(const Vector& _W3){
    this->m_W3 = _W3;
}

void cpu::NeuralNetwork::y(const double& _y){
    this->m_y = _y;
}

double& cpu::NeuralNetwork::a3(){
    return this->m_a3;
}

const cpu::Vector& cpu::NeuralNetwork::dLdW3() const{
    return m_dLdW3;
}

const cpu::Matrix& cpu::NeuralNetwork::dLdW2() const{
    return this->m_dLdW2;
}

const cpu::Matrix& cpu::NeuralNetwork::dLdW1() const{
    return this->m_dLdW1;
}