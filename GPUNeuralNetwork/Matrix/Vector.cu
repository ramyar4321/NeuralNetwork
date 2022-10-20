#include "Vector.cuh"
#include "Matrix.cuh"
#include <curand.h>
#include <iostream>

//================================//
// Constructors.
//================================//

/**
 * Constructor for Vector object 
 * when size of vector specified.
 */
gpu::Vector::Vector(int size):
                    m_size(size)
{
    this->allocateMemHost();
    this->allocateMemDevice();
}

/**
 * Constructor for Vector object with std::vector.
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
 * Copy constructor for Vector class.
*/
gpu::Vector::Vector(const gpu::Vector& other):
                m_size(other.getSize()),
                h_vec(other.h_vec),
                d_vec(other.d_vec)
{}

//================================//
// Memeory management
//================================//

/**
 *
 * Allocate vector on host.
 * Initialize all elements to zero.
 * 
*/
void gpu::Vector::allocateMemHost(){
    this->h_vec = std::shared_ptr<float>(new float[this->m_size]{0},
                                        [&](float* ptr){ delete[] ptr; });
}

/**
 * Allocate memeory space for vector on device. 
*/
void gpu::Vector::allocateMemDevice(){
    this->d_vec = std::shared_ptr<float>(nullptr,  [&](float* ptr){ cudaFree(ptr);});
    cudaMalloc((void**) &this->d_vec, this->m_size*sizeof(float));
}

/**
 * Copy vector from host to device.
 */
void gpu::Vector::copyHostToDevice(){
    cudaMemcpy(this->d_vec.get(), this->h_vec.get(), this->m_size*sizeof(float), cudaMemcpyHostToDevice);
}

/**
 * Copy vector from device to host.
 */
void gpu::Vector::copyDeviceToHost(){
    cudaMemcpy(this->h_vec.get(), this->d_vec.get(), this->m_size*sizeof(float), cudaMemcpyDeviceToHost);
}

//================================//
// CUDA kernels
//================================//

/**
 * CUDA kernel used in computing the dot 
 * product between two vectors.
 * 
 * @param res A scalar value used to store the result of the dot product
 * @param lhsVec Left hand side vector in the dot prouct
 * @param rhsVec Right hand side vector involved in the dot product
 * @param vec_size The size of either one of the vectors. 
 *                 Both vectors are assumed to be the same size.
 * 
 */
__global__ void kDot(float* res, float* lhsVec, float* rhsVec, int vec_size) {

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    float temp = 0.0f;

    if(idx < vec_size){
        temp = lhsVec[idx]*rhsVec[idx]; 
    }

    atomicAdd(res, temp);
}

/**
 * CUDA kernel used in computing the tensor product 
 * between two vectors.
 * 
 * @param res A matrix used to store the result of the tensor product
 * @param lhsVec Left hand side vector used in the tensor product
 * @param rhsVec Right hand side vector used in the tensor product
 * @param res_num_rows The number of rows of the res matrix.
 *                      It is equal to the size of right hand side vec vector.
 * @param res_num_cols The number of columns of the res matrix.
 *                      It is equal to the size of left hand side vector.
 * 
 */
__global__ void kTensor(float* res, float* lhsVec, float* rhsVec, 
                        int res_num_rows, int res_num_cols){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;

    if(idx < res_num_cols && idy < res_num_rows){
        res[idy*res_num_cols + idx] = lhsVec[idx]*rhsVec[idy];
    }

}

/**
 * CUDA kernel used to multiplying a vector by a scalar value.
 * 
 * @param res A vector used to store the result of multiplying 
 *            a vector by a scalar.
 * @param vec The vector in vector-scalar multiplication
 * @param scalar The scalar in vector-scalar multiplication
 * @param res_size The size of the result vector.
 * 
 */
__global__ void kVecScalarMult(float* res, float* vec, float scalar, int res_size){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < res_size){
        res[idx] = vec[idx]*scalar;
    }
}

/**
 * CUDA kernel used in elementwise subtraction between
 * two vectors. 
 * 
 * @param lhsVec Left hand side vector involved in vector subtraction.
 *               The result is stored in lhsVec.
 * @param rhsVec Right hand side vector invloved in vector subtraction.
 * @param vec_size The size of either one of the vectors.
 * 
 */
__global__ void kVecVecSub(float* lhsVec, float* rhsVec, int vec_size){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < vec_size){
        lhsVec[idx] -= rhsVec[idx];
    }
}

/**
 * CUDA kernel used in multiplying corresponding elements between two vectors.
 *
 * @param lhsVec Left hand side vector involved in elementwise vector multiplication
 * @param rhsVec Right hand side vector involved in elementwise vector multiplication
 * @param vec_size Size of either one of the vectors
 * 
*/
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
 * 
 * Compute the dot product between two vectors.
 * 
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
 * Compute the tensor product between two vectors.
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
 * 
 * Perform deep copy of vector.
 * 
 * That is, set the size of this vector
 * to the size of rhs vector. Then copy all
 * corresponding elements of rhs vector to this vector.
 * 
*/
void gpu::Vector::deepCopy(gpu::Vector& rhs){
    this->m_size = rhs.getSize();

    rhs.copyDeviceToHost();

    for(int j= 0 ; j < this->m_size; j++){
        this->h_vec.get()[j] = rhs[j];
    }

    this->copyHostToDevice();
}

/**
 * Overload assignment operator in order to 
 * allow assignment of another vector.
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
 * Overload assignment operator in order to allow
 * assignment of std::vector. 
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
 * Overload multiplication operator without assignment
 * to allow mutliplcation of vector with a scalar value.
 * 
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

/**
 * Overload multiplcation operator with assignment
 * in order to allow elementwise multiplcation
 * between two vectors. 
*/
gpu::Vector& gpu::Vector::operator*=( const gpu::Vector& rhs){

    int threads = 32;
    int blocks = (this->getSize() + threads - 1)/threads;

    kVecVecElementwiseMult<<<blocks, threads>>>(this->d_vec.get(), rhs.d_vec.get(), this->getSize());
    cudaDeviceSynchronize();

    return *this;

}

/**
 * Overload subtraction operator with assignment
 * in order to allow subtraction between two vectors.
 * 
*/
gpu::Vector& gpu::Vector::operator-=(const Vector& rhs){

    int threads = 32;
    int blocks = (this->getSize() + threads -1)/threads;

    kVecVecSub<<<blocks, threads>>>(this->d_vec.get(), rhs.d_vec.get(), this->getSize());
    cudaDeviceSynchronize();

    return *this;

}

/**
 * Return size of this vector.
 */
int gpu::Vector::getSize() const{
    return this->m_size;
}