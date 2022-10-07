#include "HiddenLayer.cuh"

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
// Matrix operations

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
 * 
 * TODO
 * 
 */
__device__ float reluPrime(int index, float* z){

    float f_prime = (float)(z[index] >= 0);

    return f_prime;

}

/**
 * This methode multiples a matrix with another vector.
 * 
 * TODO
 * 
 */
__global__ void kMatrixVectorMult(float* z, float* W, float* a, int W_num_cols){
    float temp = 0;

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < W_num_cols)
        for(int i=0; i < W_num_cols; i++){
            temp += W[idx*W_num_cols + i]*a[i];
        }
    z[idx] = temp;

}

/**
 * 
 * This methode performs the following operations. 
 * A matrix is first transposed, and then multipled against 
 * a vector. The resulting vector is then multipled by another vector. 
 * 
 * TODO
 * 
 */
__global__ void kMatrixTransposeVectorMult(float* delta, float* W, float* delta_,
                                            float* z, int W_num_rows, int W_num_cols){
    
    float temp;

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < W_num_cols){
        for(int k=0; k < W_num_rows; k++){
            temp += W[idx*W_num_rows + k]*delta_[k];
        }
        delta[idx] = temp*reluPrime(idx, z);
    }

}

/**
 * 
 * This methode performs the following operations.
 * A matrix is multiplied against a vector. 
 * The result is then multiped against another vector.
 * 
 * TODO
 * 
 */
__global__ void kMatrixVectorMult(float* delta, float*W, float delta_, 
                                    float* z, int delta_size){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < delta_size){
        delta[idx] = W[idx]*delta_;
        delta[idx] *= reluPrime(idx, z);
    }
}

/**
 * This methode produces a matrix by computing the tensor between two vectors.
 * 
 * TODO
 * 
 */
__global__ void kTensor(float* dLdW, float* a, float* delta, 
                        int dLdW_num_rows, int dLdW_num_cols){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;

    if(idx < dLdW_num_cols && idy < dLdW_num_rows){
        dLdW[idy*dLdW_num_cols + idx] = a[idx]*delta[idy];
    }

}

/**
 * This methode mutiples a matrix with a scalar.
 * The result is then subtracted from the another matrix.
 * 
 * TODO
 * 
 */
__global__ void kMatrixScalarMultSub(float* W, float* dLdW, float alpha,
                                      int W_num_rows, int W_num_cols){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;

    if(idx < W_num_cols && idy < W_num_rows){
        W[idy*W_num_cols + idx] -= dLdW[idy*W_num_cols+idx]*alpha;
    }

}

__global__ void kReluActivation(float* a, float* z, int a_size){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < a_size){
        a[idx] = fmaxf(0.0, z[idx]);
    }
}

/*=======================*/
// Methodes for forward propegation

/**
 * Initialize the wieghts of this hidden layer.
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

    int threads =32;
    int blocks = (this->m_z.getSize() + threads -1)/threads;

    kMatrixVectorMult<<<blocks, threads>>>(this->m_z.d_vec.get(), this->m_W.d_mat.get(), 
                                            a.d_vec.get(), this->m_W.get_num_cols());
    cudaDeviceSynchronize();

}

/**
 * Compute the activation of each neuron j in layer J using the ReLu activation function. 
 * The activation for each neuron can be computed as follows 
 * @f$z_j = max(0, z_j)$ where @f$z_j$ is the output of neuron j in layer J.
 * 
 */
void gpu::HiddenLayer::reluActivation()
{

    int threads = 32;
    int blocks = (this->m_a.getSize() + threads -1)/ threads;
    

    kReluActivation<<<blocks, threads>>>(this->m_a.d_vec.get(), this->m_z.d_vec.get(), this->m_a.getSize());
    cudaDeviceSynchronize();
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

    int threads = 32;
    int blocks = (this->m_delta.getSize() + threads - 1)/threads;

    kMatrixVectorMult<<<blocks, threads>>>(this->m_delta.d_vec.get(), W.d_vec.get(), delta,
                                            this->m_z.d_vec.get(), this->m_delta.getSize());
    cudaDeviceSynchronize();

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

    int threads = 32;
    int blocks = (this->m_delta.getSize() + threads -1)/threads;
    
    kMatrixTransposeVectorMult<<<blocks, threads>>>(this->m_delta.d_vec.get(), W.d_mat.get(), 
                                                    delta.d_vec.get(), this->m_z.d_vec.get(), 
                                                    W.get_num_rows(), W.get_num_cols());
    cudaDeviceSynchronize();

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

    int t = 32;
    int bx = (this->m_dLdW.get_num_cols() + t - 1)/t;
    int by = (this->m_dLdW.get_num_rows() + t - 1)/t;

    dim3 threads(t,t);
    dim3 blocks(bx, by);

    kTensor<<<blocks, threads>>>(this->m_dLdW.d_mat.get(), a.d_vec.get(), this->m_delta.d_vec.get(), 
                                  this->m_dLdW.get_num_rows(), this->m_dLdW.get_num_cols());
    cudaDeviceSynchronize();
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

    int t = 32;
    int bx = (this->m_dLdW.get_num_cols() + t - 1)/t;
    int by = (this->m_dLdW.get_num_rows() + t - 1)/t;

    dim3 threads(t,t);
    dim3 blocks(bx, by);

    kMatrixScalarMultSub<<<blocks, threads>>>(this->m_W.d_mat.get(), this->m_dLdW.d_mat.get(), alpha,
                                              this->m_W.get_num_rows(), this->m_W.get_num_cols());
    cudaDeviceSynchronize();

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