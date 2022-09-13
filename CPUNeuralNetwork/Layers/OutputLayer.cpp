#include "OutputLayer.hpp"
#include <cmath>
#include <iostream>

/*=======================*/
// Constructor
/**
 * Constructor for output layer J.
 * 
 * @param layerI_size The number of neurons in layer I 
 *                    where layer I is the previous layer to the output layer.
 * @param layerJ_size The number of neurons in the output layer J.
 */
cpu::OutputLayer::OutputLayer(int layerI_size, int layerJ_size):
                                bceLoss(),
                                m_z(layerJ_size, 0.0f),
                                m_a(layerJ_size, 0.0f),
                                m_delta(layerJ_size, 0.0f),
                                m_W(layerJ_size, layerI_size),
                                m_dLdW(layerJ_size, layerI_size) 
{}


/*=======================*/
// Methodes for forward propagation

/**
 * Initialize the weights of output layer.
 */
void cpu::OutputLayer::weightInitialization(){
    this->m_W.matrixInitialization();
}

/**
 * 
 * For layers I and J where layer I is the previous layer to layer J,
 * compute the output of each neuron j in layer J. 
 * The output for each neuron can be computed as follows 
 * @f$z_j = \sum_{i}^I w_{ji} a_i$ where @f$a_i$ is the output of neuron i,
 * from the previous layer I.
 * 
 * @param a The vector that contains the activation 
 *          of each neuron i of the previous layer I.
 * 
 */
void cpu::OutputLayer::computeOutput(const cpu::Vector &a)
{

    this->m_z = this->m_W*a;

}

/**
 * Compute the sigmoid activation of each neuron in layer J.
 * @f$\sigma (z_j) = \frac{1}{1+ \exp (- z_j)} = \frac{\exp (z_j)}{1+ \exp ( z_j)}$. 
 * If z_j is positive and its magnitude large, it can cause overflow when computing @f$\exp ( z_j)$ and
 * if z_j is negative and its magnitude is too large, it can cause overflow when computing @f$\exp (- z_j)$.
 * In order to avoid numerical instability, let @f$\sigma (z_j) = \frac{1}{1+ \exp (- z_j)}$ for @f$ z_j >=0 $
 * and let @f$\sigma (z_j) = \frac{\exp (z_j)}{1+ \exp ( z_j)}$ for @f$ z_j < 0 $.
 * 
 * @see https://stackoverflow.com/questions/41800604/need-help-understanding-the-caffe-code-for-sigmoidcrossentropylosslayer-for-mult
 * @see https://stackoverflow.com/questions/40353672/caffe-sigmoidcrossentropyloss-layer-loss-function
 * 
 * 
 * 
 */
void cpu::OutputLayer::computeActivation()
{
    for(int j =0; j < m_a.getSize(); j++){
        if (this->m_z[j] >= 0.0f) {
            this->m_a[j] = 1.0f / (1.0f + std::exp(-this->m_z[j]));
        } else {
            this->m_a[j] = std::exp(this->m_z[j]) / (1.0f + std::exp(this->m_z[j]));
        }
    }

}

/**
 * Perform forward propegation on the output layer.
 * 
 * @param a A vector contain the activations of each neuron
 *          in the previous layer I.
 * 
 * @return A vector containing the activation of each neuron in the output layer J.
 * 
 */
cpu::Vector cpu::OutputLayer::forwardPropegation(const cpu::Vector& a){
    this->computeOutput(a);
    this->computeActivation();

    return this->m_a;
}

/**
 * 
 * Compute the loss of the neural network using the 
 * Cross-Entropy loss function. An epsilon value is added in
 * the case that the activation is zero.
 * 
 * @see https://en.wikipedia.org/wiki/Cross_entropy
 * 
 * @param y The actual outcome from the dataset
 * 
 * @return The entropy loss 
 * 
 *
double cpu::OutputLayer::computeLoss(const cpu::Vector& y){
    double loss = 0.0f;
    // Use epsilon since log of zero is undefined.
    double epsilon = 0.0001; 

    for(int j = 0; j < this->m_a.getSize(); j ++){
        loss += -y[j]*std::log(this->m_a[j] + epsilon) - (1-y[j])*std::log(1-this->m_a[j] + epsilon);
    }

    return loss;
}*/

/*=======================*/
// Methods for backward propagation


/**
 * Compute the derivative of the sigmoid activation which can be defined as
 * @f$\sigma^{'} = \sigma(1-\sigma)$. Since m_a is the output of the sigmoid function,
 * then the derivative of the sigmoid activation can be rewritten as
 * @f$\sigma^{'} = {m_a}(1-{m_a})$ equivalently @f$\sigma^{'} = {m_a}({m_a}-1)*(-1)$
 * in order to make use of the subtraction operator of the Vector class. 
 * 
 * @return The derivative of the sigmoid function
 * 
 */
cpu::Vector cpu::OutputLayer::computeActivationPrime(){

    cpu::Vector a_prime = this->m_a*(this->m_a - 1.0)*(-1);


    return a_prime;
}

/**
 * Compute the derivative of binary cross entropy loss function
 * @f$\frac{\partial L}{\partial a} = - \frac{y}{a} + \fra{1-y}{1-a}$ where
 * @f$a$ is the output of the output neuron.
 * Since division by zero can cause numerical issues, a small value, called epsilon,
 * will be added to the denominator terms.
 * 
 * @param y The actual outcome from the dataset
 * 
 * @return The derivative of the cross entropy loss function with
 *         respect to the sigmoid activation
 *
double cpu::OutputLayer::computeLossPrime(const cpu::Vector& y){

    double loss = 0.0;
    double epsilon = 0.0001;

    for(int j=0; j < this->m_a.getSize(); j++){
        loss += -(y[j]/(this->m_a[j]+epsilon)) + ((1-y[j])/(1-this->m_a[j]+epsilon));
    }

    return loss;
}*/


/**
 * Compute the error term associated with each output neuron j in the last layer J.
 * The error term is commonly referred to as delta and is defined as the following
 * @f$\delta_j = f'(z)\frac{\partial L}{\partial a} = $
 * @f$         \sigma^{'}(z) (- \frac{y}{a} + \fra{1-y}{1-a})$
 * 
 * 
 * @param y The outcomes from the dataset
 * 
 */
void cpu::OutputLayer::computeDelta(const cpu::Vector& y){

    this->m_delta = this->computeActivationPrime() * this->bceLoss.computeLoss(this->m_a, y);
}

/**
 * 
 * For the output layer J and previous layer I, 
 * the gradient can be computed as 
 * @f$dL/dw_{ji} = \delta_j a_i$ where 
 * @f$\delta_j$ is the error term associated 
 * with neuron j of the output layer J. Since in our
 * network, there is one neuron so there is only one
 * error term. 
 * @f$a_i$ is the activation of neuron i of the pervious layer I.
 * 
 * 
 * @param delta The delta or error term from the output neuron.
 * @param a     A vector contain the activations of each neuron
 *              in the previous layer I.
 * 
 */
void cpu::OutputLayer::computeGradient(const cpu::Vector& a){

    this->m_dLdW = a.tensor(this->m_delta);

}

/**
 * Perform back propagation.
 * 
 * @param a The vector containing the activation of the neurons in layer I.
 * 
 * @return A vector containing the error terms of each neuron in output layer J.
 * 
 */
cpu::Vector cpu::OutputLayer::backPropegation(const cpu::Vector& y, const cpu::Vector& a){
    this->computeDelta(y);
    this->computeGradient(a);

    return this->m_delta;
}


/*=======================*/
// Methodes for updating the weights

/**
 * 
 * 
 * For layers I and J where layer I is the previous layer to output layer J,
 * compute the gradient descent for the weights between layer I and J.
 * @f$ w_{ji} = w_{ji} - \alpha \frac{dL}{dw_{ji}}$
 * 
 * @param alpha The step size of gradient descent
 * 
 */
void cpu::OutputLayer::gradientDecent(const double& alpha){

    this->m_W -= this->m_dLdW*alpha;

}

/**
 * 
 * Update weights using gradient descent.
 * 
 * @param alpha The step size of gradient descent
 * 
 */
void cpu::OutputLayer::updateWeigths(const double& alpha){
    this->gradientDecent(alpha);

}

// Getter methods

const cpu::Vector& cpu::OutputLayer::a() const{
    return this->m_a;
}
const cpu::Matrix& cpu::OutputLayer::W() const{
    return this->m_W;
}

// Setter methods

void cpu::OutputLayer::W(const cpu::Matrix& W){
    this->m_W = W;
}