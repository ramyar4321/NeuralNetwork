#include "HiddenLayer.hpp"

/*=======================*/
// Constructor
/**
 * Constructor for a given hidden layer J in the neural network.
 * 
 * @param layerI_size The number of neurons in the previous layer I.
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
// Methodes for forward propegation

/**
 * Initialize the wieghts of this hidden layer.
 */
void cpu::HiddenLayer::weightInitialization(){
    this->m_W.matrixInitialization();
}

/**
 * Compute the output of each neuron j in layer J. 
 * The output for each neuron can be computed as follows 
 * @f$z_j = \sum_{i}^I w_{ji} a_i$ where @f$a_i$ is the output of neuron i
 * from the pervious layer I.
 * 
 * @param a The vector that contains the activations of each neuron in layer I
 * 
 */
void cpu::HiddenLayer::computeOutput(const cpu::Vector &a)
{

    this->m_z = this->m_W*a;

}

/**
 * Compute the activation of each neuron j in layer J using the ReLu activation function. 
 * The activation for each neuron can be computed as follows 
 * @f$z_j = max(0, z_j)$ where @f$z_j$ is the output of neuron j in layer J.
 * 
 */
void cpu::HiddenLayer::reluActivation()
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
 * Perform forward propegation on the output layer.
 * 
 * @param a A vector contain the activations of each neuron
 *          in the previous layer.
 * 
 */
void cpu::HiddenLayer::forwardPropegation(const cpu::Vector& a){
    this->computeOutput(a);
    this->reluActivation();
}

/*=======================*/
// Methodes for backward propegation


/**
 * Compute activation of neuron j of layer J using the derivative of ReLu
 * activation function.
 * @f$\begin{math}
        f'(z_j)=\left\{
            \begin{array}{ll}
                0, & \mbox{if $x<0$}.\\
                1, & \mbox{if $x>0$}.
            \end{array}
        \right.
    \end{math}$
 * The derivative is undefined at z_j = 0 but it can be set to zero
 * in order to produce sparse vector.
 * 
 * @param z A vector that contains the output of each neuron in layer I
 * 
 * @return A vector containing @f$f'(z_j)$ where @f$z_j$ 
 *         is the output of neuron j of layer J 
 *         and f' is the derivative of the relu activation function.
 */
cpu::Vector cpu::HiddenLayer::reluPrime(){
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
 * For layers J < K, compute the error term associated with each neuron j of layer J.
 * @f$\delta_j = f'(z_j)\sum_{k=0}^{n_K} w_{kj} \delta_k$ where
 * @f$f'$ is the derivative of the ReLu activation function,
 * @f$z_j$ is the output of neuron j of layer J, @f$n_K$
 * is the number of neurons in layer K, @f$w_{ji}$ is the
 * weight from neuron j of layer J to neuron k of layer K,
 * and @f$\delta_k$ is the error term of neuron k of layer K.
 * 
 * 
 * @param W A vector containing the weigths between layer J and K
 * @param delta The error terms of each neuron k of layer K.
 * 
 */
void cpu::HiddenLayer::computeDelta(const cpu::Vector& W, const double& delta){

    cpu::Vector f_prime = this->reluPrime();

    this->m_delta = W*delta;
    this->m_delta *= f_prime;

}

/**
 * For layers J < K, compute the error term associated with each neuron j of layer J.
 * @f$\delta_j = f'(z_j)\sum_{k=0}^{n_K} w_{kj} \delta_k$ where
 * @f$f'$ is the derivative of the ReLu activation function,
 * @f$z_j$ is the output of neuron j of layer J, @f$n_K$
 * is the number of neurons in layer K, @f$w_{ji}$ is the
 * weight from neuron j of layer J to neuron k of layer K,
 * and @f$\delta_k$ is the error term of neuron k of layer K.
 * 
 * @param W A matrix containing the weigths between layer J and K
 * @param delta_ A vector cotaining the error terms of each neuron k of layer K
 * 
 */
void cpu::HiddenLayer::computeDelta(const cpu::Matrix& W, 
                                    const cpu::Vector& delta_){
    cpu::Vector f_prime = this->reluPrime();

    cpu::Matrix W_tranpose = W.transpose();
    this->m_delta = W_tranpose*delta_;

    this->m_delta *= f_prime;

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
 * 
 */
void cpu::HiddenLayer::computeGradient(const cpu::Vector& a){

    this->m_dLdW = a.tensor(this->m_delta);
}

/**
 * Perform Back propegation.
 * This methode is called to perform back propegation when this hidden layer
 * is the second hidden layer in the neural network.
 */
void cpu::HiddenLayer::backPropegation(const cpu::Vector& W, const double& delta, const cpu::Vector& a){
    this->computeDelta(W, delta);
    this->computeGradient(a);
}

/**
 * Perform back propegation/
 * This mehtode is called to perform back propegation when this 
 * hidden layer is the first hidden layer in the neural network.
 */
void cpu::HiddenLayer::backPropegation(const cpu::Matrix& W, const cpu::Vector& delta, const cpu::Vector& a){
    this->computeDelta(W, delta);
    this->computeGradient(a);
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
void cpu::HiddenLayer::gradientDecent(const double& alpha){

    this->m_W -= this->m_dLdW*alpha;

}

/**
 * 
 * Update weights using gradient descent.
 * 
 */
void cpu::HiddenLayer::updateWeigths(const double& alpha){
    this->gradientDecent(alpha);

}