#include "HiddenLayer.cuh"
#include "../ErrorHandling/CudaError.cuh"

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
                                m_z(layerJ_size),
                                m_a(layerJ_size),
                                m_delta(layerJ_size),
                                m_W(layerJ_size, layerI_size),
                                m_dLdW(layerJ_size, layerI_size)              
{}

/*=======================*/
// CUDA kernels

/**
 * CUDA kernel that computes the elements of vector
 * containing f' for neurons of a given hidden layer J.
 * 
 * @param f_prime Vector containing f' for neurons of given hidden layer J
 * @param z       Vector cotnaining the output of neurons of given hideen layer J
 * @param z_size  Size of vector z. Assumed to be the equal to size of f_prime.
 * 
 * 
*/
__global__ void kReluPrime(float* f_prime, float* z, float z_size){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < z_size){
        f_prime[idx] = (float)(z[idx] >= 0);
    }

}


/**
 * CUDA kernel that computes the elements of a vector 
 * containing the activations for neurons of given hidden layer J.
 * 
 * @param a Vector containing the ReLu activation of neurons in hidden layer J.
 * @param z Vector containing input of neurons in hidden layer J.
 * @param a_size Size of a vector a. Assumed to be equal to size of z.
 *  
*/
__global__ void kReluActivation(float* a, float* z, int a_size){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < a_size){
        a[idx] = fmaxf(0.0, z[idx]);
    }
}

/*=======================*/
// Methodes for forward propegation

/**
 * Initialize the weights of this hidden layer.
 */
void gpu::HiddenLayer::weightInitialization(){
    this->m_W.matrixInitializationDevice();
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

    this->m_z = this->m_W*a;

}

/**
 * Compute the activation of each neuron j in layer J using the ReLu activation function. 
 * The activation for each neuron can be computed as follows 
 * @f$a_j = max(0, z_j)$ where @f$z_j$ is the output of neuron j in layer J.
 * 
 */
void gpu::HiddenLayer::reluActivation()
{

    int threads = 32;
    int blocks = (this->m_a.getSize() + threads -1)/ threads;
    std::string err_msg = "cudaDeviceSynchronize failed in HiddenLayer reluActivation.";

    kReluActivation<<<blocks, threads>>>(this->m_a.d_vec.get(), this->m_z.d_vec.get(), this->m_a.getSize());
    gpu::CudaError::checkCudaError(cudaDeviceSynchronize(), err_msg);
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
 * 
 */
gpu::Vector gpu::HiddenLayer::reluPrime(){
    gpu::Vector f_prime(this->m_z.getSize());

    int threads = 32;
    int blocks = (this->m_z.getSize() + threads -1)/threads;
    
    std::string err_msg = "cudaDeviceSynchronize failed in HiddenLayer reluPrime.";

    kReluPrime<<<blocks, threads>>>(f_prime.d_vec.get(), this->m_z.d_vec.get(), this->m_z.getSize());
    gpu::CudaError::checkCudaError(cudaDeviceSynchronize(), err_msg);

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
 * @param delta The error terms of neuron k of layer K.
 * 
 */
void gpu::HiddenLayer::computeDelta(const gpu::Vector& W, const float& delta){

    gpu::Vector f_prime = this->reluPrime();

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
void gpu::HiddenLayer::computeDelta(const gpu::Matrix& W, 
                                    const gpu::Vector& delta){

    gpu::Vector f_prime = this->reluPrime();

    gpu::Matrix W_transpose = W.transpose();
    this->m_delta = W_transpose*delta;

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
void gpu::HiddenLayer::computeGradient(const gpu::Vector& a){

    this->m_dLdW = a.tensor(this->m_delta);
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
 * Perform gradient descent. For any given weight between layers I < J,
 * the weight can be updated using the following.
 * @f$ w_{ji} = w_{ji} - \alpha \frac{dL}{dw_{ji}}$
 * 
 * @param alpha The step size of gradient descent
 * 
 */
void gpu::HiddenLayer::gradientDecent(const float& alpha){

    this->m_W -= this->m_dLdW*alpha;

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

void gpu::HiddenLayer::WDeepCopy(gpu::Matrix& W){
    this->m_W.deepCopy(W);
}