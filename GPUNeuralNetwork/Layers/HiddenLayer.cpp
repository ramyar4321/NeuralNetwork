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
gpu::HiddenLayer::HiddenLayer(int layerI_size, int layerJ_size):
                                m_z(layerJ_size, 0.0f),
                                m_a(layerJ_size, 0.0f),
                                m_delta(layerJ_size, 0.0f),
                                m_W(layerJ_size, layerI_size),
                                m_dLdW(layerJ_size, layerI_size)              
{}

/*=======================*/
// Matrix operations

/**
 * This methode multiples a matrix with another vector.
 * 
 */
void gpu::HiddenLayer::matrixVectorMult(gpu::Vector& z, const gpu::Matrix& W, const gpu::Vector& a){
    float temp;
    
    for (int j=0; j < W.get_num_rows(); j++){
        temp = 0;
        for(int i=0; i < W.get_num_cols(); i++){
            temp += W(j,i)*a[i];
        }
        z[j] = temp;
    }
}

/**
 * 
 * This methode performs the following operations. 
 * A matrix is first transposed, and then multipled against 
 * a vector. The resulting vector is then multipled by another vector. 
 * 
 */
void gpu::HiddenLayer::matrixTransposeVectorMult(gpu::Vector& delta, const gpu::Matrix& W, 
                                                 const gpu::Vector& delta_, const gpu::Vector& z){
    
    float temp;

    int layerJ_size = W.get_num_cols();
    int layerK_size = W.get_num_rows();

    for(int j=0; j < layerJ_size; j++){
        temp = 0.0;
        for(int k=0; k < layerK_size; k++){
            temp += W(k,j)*delta_[k];
        }
        delta[j] = temp*reluPrime(z[j]);
        
    }
}

/**
 * 
 * This methode performs the following operations.
 * A matrix is multiplied against a vector. 
 * The result is then multiped against another vector.
 * 
 */
void gpu::HiddenLayer::matrixVectorMult(gpu::Vector& delta, const gpu::Vector& W, 
                                        const float& delta_, const gpu::Vector& z){
    for (int j=0; j < W.getSize(); j++){
        delta[j] = W[j]*delta_;
        delta[j] *= reluPrime(z[j]);
    }
}

/**
 * This methode produces a matrix by computing the tensor between two vectors.
 */
void gpu::HiddenLayer::tensor(gpu::Matrix& W, const gpu::Vector& a, const gpu::Vector& delta){

    for(int j=0; j < W.get_num_rows(); j++){
        for(int i=0; i < W.get_num_cols(); i++){
            W(j,i) = a[i]*delta[j];
        }
    }
}

/**
 * This methode mutiples a matrix with a scalar.
 * The result is then subtracted from the another matrix.
 */
void gpu::HiddenLayer::matrixScalarMultSub(gpu::Matrix& W, const gpu::Matrix& dLdW, const float& alpha){
    for(int j=0; j < W.get_num_rows(); j++){
        for(int i=0; i < W.get_num_cols(); i++){
            W(j,i) -= dLdW(j,i)*alpha;
        }
    }
}

/*=======================*/
// Methodes for forward propegation

/**
 * Initialize the wieghts of this hidden layer.
 */
void gpu::HiddenLayer::weightInitialization(){
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
void gpu::HiddenLayer::computeOutput(const gpu::Vector &a)
{
    this->matrixVectorMult(this->m_z, this->m_W, a);

}

/**
 * Compute the activation of each neuron j in layer J using the ReLu activation function. 
 * The activation for each neuron can be computed as follows 
 * @f$z_j = max(0, z_j)$ where @f$z_j$ is the output of neuron j in layer J.
 * 
 */
void gpu::HiddenLayer::reluActivation()
{

    for (int j=0; j<this->m_z.getSize(); j++) {
        this->m_a[j] = std::max(0.0f, this->m_z[j]);
        /*if(this->m_z[j] > 0.0f ){
            this->m_a[j] = this->m_z[j];
        }else{
            this->m_a[j] = 0.0f;
        }*/
    } 
}

/**
 * Perform forward propegation on this hidden layer J.
 * 
 * @param a A vector contain the activations of each neuron
 *          in the previous layer.
 * 
 * @return A vector containing the activation of the neurons in 
 *         this hidden layer J.
 * 
 */
gpu::Vector gpu::HiddenLayer::forwardPropegation(const gpu::Vector& a){
    this->computeOutput(a);
    this->reluActivation();

    return this->m_a;
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
 * @param z A value that contains the output of a given neuron i in layer I
 * 
 * @return @f$f'(z_j)$ where @f$z_j$ 
 *         is the output of neuron j of layer J 
 *         and f' is the derivative of the relu activation function.
 */
float gpu::HiddenLayer::reluPrime(const float& z){

    float fprime = static_cast<float>(z >= 0);

    return fprime;
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
void gpu::HiddenLayer::computeDelta(const gpu::Vector& W, const float& delta){

    this->matrixVectorMult(this->m_delta, W, delta, this->m_z);

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
void gpu::HiddenLayer::computeDelta(const gpu::Matrix& W, 
                                    const gpu::Vector& delta){

    
    this->matrixTransposeVectorMult(this->m_delta, W, delta, this->m_z);

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
void gpu::HiddenLayer::computeGradient(const gpu::Vector& a){

    this->tensor(this->m_dLdW, a, this->m_delta);
}

/**
 * Perform Back propegation.
 * 
 * This methode is called to perform back propegation when this hidden layer
 * is the second hidden layer in the neural network.
 * 
 * @return A vector containing the error term for each neuron of this hidden layer.
 */
gpu::Vector gpu::HiddenLayer::backPropegation(const gpu::Vector& W, const float& delta, const gpu::Vector& a){
    this->computeDelta(W, delta);
    this->computeGradient(a);

    return this->m_delta;
}

/**
 * Perform back propegation.
 * 
 * This methode is called to perform back propegation when this 
 * hidden layer is not the last hidden layer in the neural network.
 * 
 * @return A vector containing the error terms for all neurons 
 *         of this hidden layer.
 * 
 */
gpu::Vector gpu::HiddenLayer::backPropegation(const gpu::Matrix& W, const gpu::Vector& delta, const gpu::Vector& a){
    this->computeDelta(W, delta);
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
void gpu::HiddenLayer::gradientDecent(const float& alpha){

    this->matrixScalarMultSub(this->m_W, this->m_dLdW, alpha);

}

/**
 * 
 * Update weights using gradient descent.
 * 
 */
void gpu::HiddenLayer::updateWeigths(const float& alpha){
    this->gradientDecent(alpha);

}

/*=======================*/

// Getter methods
            
const gpu::Vector& gpu::HiddenLayer::a() const{
    return this->m_a;
}


const gpu::Matrix& gpu::HiddenLayer::W() const{
    return this->m_W;
}

const gpu::Matrix& gpu::HiddenLayer::dLdW() const{
    return this->m_dLdW;
}

// Setter methods

void gpu::HiddenLayer::W(const gpu::Matrix& W){
    this->m_W = W;
}

void gpu::HiddenLayer::WDeepCopy(const gpu::Matrix& W){
    this->m_W.deepCopy(W);
}