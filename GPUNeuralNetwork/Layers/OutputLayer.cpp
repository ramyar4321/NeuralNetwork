#include "OutputLayer.hpp"
#include <cmath>
#include <iostream>

/*=======================*/
// Constructor
/**
 * Constructor for output layer J.
 * 
 * @param layerJ_size The number of neurons in layer I 
 *                    where layer I is the previous layer to the output layer.
 */
gpu::OutputLayer::OutputLayer(int layerI_size):
                                m_z(0.0),
                                m_a(0.0),
                                m_W(layerI_size, 0.0),
                                m_dLdW(layerI_size, 0.0)
{}

/*=======================*/

/**
 * Compute the dot product between two vectors.
 * 
 * 
 */
void gpu::OutputLayer::dot(float& z, const gpu::Vector& W, const gpu::Vector& a) {
    float temp = 0;

    for(int j=0; j <this->m_W.getSize(); j++){
        temp += W[j]*a[j]; 
    }

    z = temp;
}

/**
 * Compute the vector multiplication between a vector and a scalar value.
 * 
 * 
 */
void gpu::OutputLayer::vecScalarMult(gpu::Vector& dLdW, const gpu::Vector& a, const float& delta){


    for(int j=0; j < dLdW.getSize(); j++){
        dLdW[j] = a[j]*delta;
    }
}

/**
 * 
 * This methode mutiplies a vector by a scalar.
 * The resulting vector is then subtracted from a another vector.
 * 
 */
void gpu::OutputLayer::vecScalarMultSub(gpu::Vector& W, const gpu::Vector& dLdW, const float& alpha){


    for(int j=0; j < W.getSize(); j++){
        W[j] -= dLdW[j]*alpha;
    }
}

/*=======================*/
// Methodes for forward propegation

/**
 * Initialize the wieghts of this hidden layer.
 */
void gpu::OutputLayer::weightInitialization(){
    this->m_W.vectorInitialization();
}

/**
 * Compute the output of each neuron j in output layer J. 
 * The output for each neuron can be computed as follows 
 * @f$z_j = \sum_{i}^I w_{ji} a_i$ where @f$a_i$ is the output of neuron i
 * from the pervious layer I.
 * 
 * @param a The vector that contains the activations of each neuron in layer I
 * 
 */
void gpu::OutputLayer::computeOutput(const gpu::Vector& a)
{

    this->dot(this->m_z, this->m_W, a);
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
 * 
 */
void gpu::OutputLayer::sigmoidActivation()
{
    if (this->m_z >= 0.0f) {
        this->m_a = 1.0f / (1.0f + std::exp(-this->m_z));
    } else {
        this->m_a = std::exp(this->m_z) / (1.0f + std::exp(this->m_z));
    }

}

/**
 * Perform forward propegation on the output layer.
 * 
 * @param a A vector contain the activations of each neuron
 *          in the previous layer.
 * 
 * @return The activation of the output neuron.
 * 
 */
float gpu::OutputLayer::forwardPropegation(const gpu::Vector& a){
    this->computeOutput(a);
    this->sigmoidActivation();

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
 */
float gpu::OutputLayer::bceLoss(const float &y){
    float loss = 0.0f;
    // Use epsilon since log of zero is undefined.
    double epsilon = 0.0001; 


    loss += -y*std::log(this->m_a + epsilon) - (1-y)*std::log(1-this->m_a + epsilon);

    return loss;
}

/*=======================*/
// Methodes for backward propegation


/**
 * Compute the derivative of the sigmoid activiation which can be defined as
 * @f$\sigma^{'} = \sigma(1-\sigma)$. SInce m_a is the output of the sigmoid function,
 * then the derivative of the sigmoid activiation can be rewritten as
 * @f$\sigma^{'} = {m_a}(1-{m_a})$
 * 
 * @return The derivative of the sigmoid function
 * 
 */
float gpu::OutputLayer::sigmoidPrime(){

    float a_prime = this->m_a*(1.0 - this->m_a);


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
 *         respect to the sigmoid activation a
 */
float gpu::OutputLayer::bceLossPrime(const float &y){

    float loss = 0.0;
    double epsilon = 0.0001;

    loss += -(y/(this->m_a+epsilon)) + ((1-y)/(1-this->m_a+epsilon));

    return loss;
}


/**
 * Compute the error term associated with the output neuron j in the last layer.
 * The error term is commonly referred to as delta and is defined as the following
 * @f$\delta_j = f'(z)\frac{\partial L}{\partial a} = $
 * @f$         \sigma^{'}(z) (- \frac{y}{a} + \fra{1-y}{1-a})$
 * 
 * 
 * @param y The outcomes from the dataset
 * 
 */
void gpu::OutputLayer::computeDelta(const float& y){

    this->m_delta = this->sigmoidPrime() * this->bceLossPrime(y);
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
void gpu::OutputLayer::computeGradient(const gpu::Vector& a){

    this->vecScalarMult(this->m_dLdW, a, this->m_delta);

}

/**
 * Perform back propegation.
 * 
 * @return The error term associated with this output neuron.
 * 
 */
float gpu::OutputLayer::backPropegation(const float& y, const gpu::Vector& a){
    this->computeDelta(y);
    this->computeGradient(a);

    return this->m_delta;
}

/*=======================*/
// Methodes for updating the weights

/**
 * 
 * 
 * Perform gradient descent. For any given weight between layers I < J
 * where Iis the previous layer and J is the output layer,
 * the weight can be updated using the following.
 * @f$ w_{ji} = w_{ji} - \alpha \frac{dL}{dw_{ji}}$
 * 
 * @param alpha The step size of gradient descent
 * 
 */
void gpu::OutputLayer::gradientDecent(const float& alpha){

    this->vecScalarMultSub(this->m_W, this->m_dLdW, alpha);

}

/**
 * 
 * Update weights using gradient descent.
 * 
 */
void gpu::OutputLayer::updateWeigths(const float& alpha){
    this->gradientDecent(alpha);

}

/*=======================*/
// Setter and getter methods

// Getter methods

const float& gpu::OutputLayer::a() const{
    return this->m_a;
}


const gpu::Vector& gpu::OutputLayer::W() const{
    return this->m_W;
}

const gpu::Vector& gpu::OutputLayer::dLdW() const{
    return this->m_dLdW;
}

// Setter methods

void gpu::OutputLayer::W(const gpu::Vector& W){
    this->m_W = W;
}

void gpu::OutputLayer::WDeepCopy(const gpu::Vector& W){
    this->m_W.deepCopy(W);
}