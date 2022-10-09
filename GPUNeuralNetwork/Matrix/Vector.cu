#include "Vector.cuh"
#include "Matrix.cuh"
#include <curand.h>
#include <iostream>

/**
 * Constructor for Vector object with size of vector and
 * initial values for each element are specified.
 */
gpu::Vector::Vector(int size):
                    m_size(size)
{
    this->allocateMemHost();
    this->allocateMemDevice();
}

/**
 * Constructor for Vector object with initializer list.
 */
gpu::Vector::Vector(std::vector<float> rhs):
                    m_size(rhs.size())
{
    this->allocateMemHost();
    this->allocateMemDevice();

    for(int j= 0; j < this->m_size; j++){
        this->h_vec.get()[j] = rhs[j];
    }

    this->copyHostToDevice();
}

/**
 * TODO
*/
gpu::Vector::Vector(const gpu::Vector& other):
                m_size(other.getSize()),
                h_vec(other.h_vec),
                d_vec(other.d_vec)
{}

/**
 * TODO
*/
void gpu::Vector::allocateMemHost(){
    this->h_vec = std::shared_ptr<float>(new float[this->m_size]{0},
                                        [&](float* ptr){ delete[] ptr; });
}

/**
 * TODO
*/
void gpu::Vector::allocateMemDevice(){
    this->d_vec = std::shared_ptr<float>(nullptr,  [&](float* ptr){ cudaFree(ptr);});
    cudaMalloc((void**) &this->d_vec, this->m_size*sizeof(float));
}

/**
 * TODO
 */
void gpu::Vector::copyHostToDevice(){
    cudaMemcpy(this->d_vec.get(), this->h_vec.get(), this->m_size*sizeof(float), cudaMemcpyHostToDevice);
}

/**
 * TODO
 */
void gpu::Vector::copyDeviceToHost(){
    cudaMemcpy(this->h_vec.get(), this->d_vec.get(), this->m_size*sizeof(float), cudaMemcpyDeviceToHost);
}

/**
 * Compute the dot product between two vectors.
 * 
 * TODO
 * 
 */
__global__ void kDot(float* z, float* W, float* a, int W_size) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    float temp = 0.0f;

    if(idx < W_size){
        temp = W[idx]*a[idx]; 
    }

    atomicAdd(z, temp);
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
 * Compute the vector multiplication between a vector and a scalar value.
 * 
 * TODO
 * 
 */
__global__ void kVecScalarMult(float* dLdW, float* a, float delta, int dLdW_size){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < dLdW_size){
        dLdW[idx] = a[idx]*delta;
    }
}

/**
 * 
 * This methode mutiplies a vector by a scalar.
 * The resulting vector is then subtracted from a another vector.
 * 
 */
__global__ void kVecScalarMultSub(float* W, float* dLdW, int W_size){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < W_size){
        W[idx] -= dLdW[idx];
    }
}

__global__ void kVecVecElementwiseMult(float* delta, float* f_prime,  int delta_size){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < delta_size){
        delta[idx] *= f_prime[idx];
    }
}

/**
 * Initialize the elements of the mvector to random values that come
 * from a Gaussian Distribtuion centered at 0 with standard deviations of 
 * @f$\sqrt{ \farc{1}{n_{I}}} $ where @f$n_{I}$ is the size of layer @f$I$.
 * 
 */
void gpu::Vector::vectorInitializationDevice()
{

    curandGenerator_t gen;
    curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 
                1234ULL);

    float mean = 0.0;
    float stddev = 1/sqrtf(1.0/(float)this->m_size);

    curandGenerateNormal(gen, this->d_vec.get(), this->m_size, mean, stddev);

    curandDestroyGenerator(gen);

}

/**
 * TODO
*/
gpu::Scalar gpu::Vector::dot(const gpu::Vector& rhs) const{
    gpu::Scalar res(0.0f);

    int threads = 32;
    int blocks = (this->getSize() + threads - 1)/threads;

    kDot<<<blocks, threads>>>(res.d_scalar.get(), this->d_vec.get(), 
                                rhs.d_vec.get(), this->getSize());
    cudaDeviceSynchronize();

    return res;

}

/**
 * TODO
*/
gpu::Matrix gpu::Vector::tensor(const Vector& rhs) const{

    int num_rows = rhs.getSize();
    int num_cols = this->m_size;
    gpu::Matrix res(num_rows, num_cols);

    int t = 32;
    int bx = (num_cols + t - 1)/t;
    int by = (num_rows + t - 1)/t;

    dim3 threads(t,t);
    dim3 blocks(bx, by);

    kTensor<<<blocks, threads>>>(res.d_mat.get(), this->d_vec.get(), rhs.d_vec.get(), 
                                  num_rows, num_cols);
    cudaDeviceSynchronize();

    return res;
}

/**
 * TODO
*/
void gpu::Vector::deepCopy(gpu::Vector& rhs){
    this->m_size = rhs.getSize();

    rhs.copyDeviceToHost();

    for(int j= 0 ; j < this->m_size; j++){
        this->h_vec.get()[j] = rhs[j];
    }

    this->copyHostToDevice();
}

void gpu::Vector::printVec(){

    this->copyDeviceToHost();

    for (int i=0; i< this->m_size; ++i) {
        std::cout << this->h_vec.get()[i] << std::endl;
    }
}

/**
 * TODO
*/
gpu::Vector& gpu::Vector::operator=(const Vector& rhs){
    // Check if object is being assigned to itself.
    if(this == &rhs){
        return *this;
    }

    this->m_size = rhs.getSize();

    this->h_vec = rhs.h_vec;
    this->d_vec = rhs.d_vec;

    return *this;

}

/**
 * TODO
*/
void gpu::Vector::operator=(const std::vector<float>& rhs){
    this->m_size = rhs.size();


    for(int j= 0 ; j < this->m_size; j++){
        this->h_vec.get()[j] = rhs[j];
    }

    this->copyHostToDevice();
}

/**
 * Overload equality operator.
 * 
 * Two vectora are equal if and only if
 * they have the same size and their
 * corresonding elements are equal.
 * 
 * return true if two vectors are equal,
 *        false otherwise
 */
bool gpu::Vector::operator==(Vector& rhs){
    bool areEqual = true;

    // Variables to store the element of vectors to be compared
    float this_val = 0.0;
    float rhs_val = 0.0;

    // Fixed error for comparison between two given values
    constexpr double epsilon = 0.01; 

    rhs.copyDeviceToHost();
    this->copyDeviceToHost();

    //Check if the sizes of the two vectors are equal
    if( this->m_size != rhs.getSize()){
            areEqual = false;
    }else{
        // Check if corresponding elements of the two vectors are equal
            for(int i = 0; i < this->m_size; i++){
                this_val = this->h_vec.get()[i];
                rhs_val = rhs[i];
                if(!(std::abs(this_val - rhs_val) < epsilon)){
                    areEqual = false;
                }
            }

    }

    return areEqual;
}

/**
 * Overload operator[] for read-only operation on elements of this Vector.
 */
const float gpu::Vector::operator[](const int &input) const{
    return h_vec.get()[input];
}

/**
 * Overload operator[] for write operation on elements of this Vector.
 */
float& gpu::Vector::operator[](const int &input) {
    return h_vec.get()[input];
}

/**
 * TODO
*/
gpu::Vector gpu::Vector::operator*(const float& rhs) const{

    gpu::Vector res(this->m_size);

    int threads = 32;
    int blocks = (this->getSize() + threads -1)/threads;

    kVecScalarMult<<<blocks, threads>>>(res.d_vec.get(), this->d_vec.get(), 
                                        rhs, this->getSize());
    cudaDeviceSynchronize();

    return res;

}

gpu::Vector& gpu::Vector::operator*=( const gpu::Vector& rhs){

    int threads = 32;
    int blocks = (this->getSize() + threads - 1)/threads;

    kVecVecElementwiseMult<<<blocks, threads>>>(this->d_vec.get(), rhs.d_vec.get(), this->getSize());
    cudaDeviceSynchronize();

    return *this;

}

/**
 * TODO
*/
gpu::Vector& gpu::Vector::operator-=(const Vector& rhs){

    int threads = 32;
    int blocks = (this->getSize() + threads -1)/threads;

    kVecScalarMultSub<<<blocks, threads>>>(this->d_vec.get(), rhs.d_vec.get(), this->getSize());
    cudaDeviceSynchronize();

    return *this;

}

/**
 * Return size of this vector.
 */
int gpu::Vector::getSize() const{
    return this->m_size;
}