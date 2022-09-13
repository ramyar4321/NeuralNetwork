#include "HiddenLayer.hpp"

/*=======================*/
// Constructor
/**
 * Constructor for a given hidden layer J in the neural network.
 * 
 * @param layerI_size The number of neurons in the previous layer, layer I.
 * @param layerJ_size The number of neurons in layer J.
 * 
 */
cpu::HiddenLayer::HiddenLayer(int layerI_size, int layerJ_size):
                                m_z(layerJ_size, 0.0f),
                                m_a(layerJ_size, 0.0f),
                                m_delta(layerJ_size, 0.0f),
                                m_W(layerJ_size, layerI_size),
                                m_dLdW(layerJ_size, layerI_size)              
{}


/*=======================*/
// Methods for forward propagation

/**
 * Initialize the weights of this hidden layer.
 */
void cpu::HiddenLayer::weightInitialization(){
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
 *          of each neuron, neuron i, of the previous layer, layer I.
 * 
 */
void cpu::HiddenLayer::computeOutput(const cpu::Vector &a)
{

    this->m_z = this->m_W*a;

}

/**
 * Compute the activation of each neuron j, of layer J
 * using the ReLu activation function. 
 * The activation for each neuron can be computed as follows 
 * @f$z_j = max(0, z_j)$ where @f$z_j$ is the output of neuron j in layer J.
 * 
 */
void cpu::HiddenLayer::computeActivation()
{

    for (int j=0; j<this->m_z.getSize(); j++) {
        if(this->m_z[j] > 0.0f ){
            this->m_a[j] = this->m_z[j];
        }else{
            this->m_a[j] = 0.0f;
        }
    } 
}

/**
 * For layers I and J where layer I is the previous layer to layer J,
 * perform forward propagation for each neuron of the layer J.
 * 
 * @param a A vector contain the activation of each neuron in layer J.
 * 
 * @return A vector containing the activation of each neuron in layer J.
 * 
 */
cpu::Vector cpu::HiddenLayer::forwardPropegation(const cpu::Vector& a){
    this->computeOutput(a);
    this->computeActivation();

    return this->m_a;
}

/*=======================*/
// Methodes for backward propegation


/**
 * Compute f' of neuron j of layer J
 * using the derivative of ReLu activation function.
 * @f$\begin{math}
        f'(z_j)= \left\{
            \begin{array}{ll}
                0, & \mbox{if $x<0$}.\\
                1, & \mbox{if $x>0$}.
            \end{array}
        \right.
    \end{math}$
 * The derivative is undefined at z_j = 0 but it can be set to zero
 * in order to produce sparse vector.
 * 
 * 
 * @return A vector containing f' for each neuron j of layer J.
 * 
 */
cpu::Vector cpu::HiddenLayer::computeActivationPrime(){
   cpu::Vector f_prime(this->m_z.getSize(), 0.0f);

    for (int i = 0; i < this->m_z.getSize(); i++){
        if(this->m_z[i] <= 0){
            f_prime[i] = 0;
        }else{
            f_prime[i] = 1;
        }
    }

    return f_prime;
}


/**
 * For layers J and K where layer K is the next layer to layer J,
 * compute the error term associated with each neuron j of layer J.
 * @f$\delta_j = f'(z_j)\sum_{k=0}^{n_K} w_{kj} \delta_k$ where
 * f' is the derivative of the ReLu activation function,
 * @f$z_j$ is the output of neuron j of layer J, @f$n_K$
 * is the number of neurons in layer K, @f$w_{ji}$ is the
 * weight from neuron j of layer J to neuron k of layer K,
 * and @f$\delta_k$ is the error term of neuron k of layer K.
 * 
 * @param W A matrix containing the weights between layer J and K
 * @param delta A vector containing the error terms of each neuron k of layer 
 * 
 */
void cpu::HiddenLayer::computeDelta(const cpu::Matrix& W, 
                                    const cpu::Vector& delta){
    cpu::Vector f_prime = this->computeActivationPrime();

    cpu::Matrix W_transpose = W.transpose();
    this->m_delta = W_transpose*delta;

    this->m_delta *= f_prime;

}

/**
 * 
 * For layer I and J where layer I is the previous layer to layer J,
 * compute the gradient for each neuron j of layer J
 * @f$\frac{dL}{dw_{ji}} = a_i * \delta_{j}$ where @f$w_{ji}$ is the weight from
 * neuron i of layer I to neuron j of layer J, @f$a_i$ is the activation of neuron
 * i of layer I, and @f$\delta_{j}$ is the error term of neuron j of layer J.
 * 
 * @param detla A vector containing the error terms for each neuron of layer J
 * @param a     A vector containing the activation of each neuron of layer I
 * 
 */
void cpu::HiddenLayer::computeGradient(const cpu::Vector& a){

    this->m_dLdW = a.tensor(this->m_delta);
}


/**
 * 
 * For layers J and K where layer K is the next layer to layer J, perform back propagation
 * 
 * @param W The matrix containing the weights between layers J and K  for J < K.
 * @param delta The vector containing the error terms for each neuron in layer K
 * @param a The vector containing the activation of the neurons in layer I.
 * 
 * @return A vector containing the error terms of each neuron in layer J.
 * 
 */
cpu::Vector cpu::HiddenLayer::backPropegation(const cpu::Matrix& W, 
                                                const cpu::Vector& delta, 
                                                const cpu::Vector& a){
    this->computeDelta(W, delta);
    this->computeGradient(a);

    return this->m_delta;
}


/*=======================*/
// Methodes for updating the weights

/**
 * 
 * For layers I and J where layer I is the previous layer to layer J,
 * compute the gradient descent for the weights between layer I and J.
 * @f$ w_{ji} = w_{ji} - \alpha \frac{dL}{dw_{ji}}$
 * 
 * @param alpha The step size of gradient descent
 * 
 */
void cpu::HiddenLayer::gradientDecent(const double& alpha){

    this->m_W -= this->m_dLdW*alpha;

}

/**
 * 
 * Update weights using gradient descent.
 * 
 * @param alpha The step size of gradient descent
 * 
 */
void cpu::HiddenLayer::updateWeigths(const double& alpha){
    this->gradientDecent(alpha);

}

// Getter methods

const cpu::Vector& cpu::HiddenLayer::a() const{
    return this->m_a;
}
const cpu::Matrix& cpu::HiddenLayer::W() const{
    return this->m_W;
}

// Setter methods

void cpu::HiddenLayer::W(const cpu::Matrix& W){
    this->m_W = W;
}