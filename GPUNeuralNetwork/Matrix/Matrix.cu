#include "Matrix.cuh"
#include "Vector.cuh"
#include <iostream>
#include <algorithm>
#include <curand.h>

//================================//
// Constructors.
//================================//

/**
 * TDOD
 */
gpu::Matrix::Matrix(int num_rows, 
                    int num_cols):
                    m_num_rows(num_rows),
                    m_num_cols(num_cols)
{
    this->allocateMemHost();
    this->allocateMemDevice();
}

/**
 * TODO
 */
gpu::Matrix::Matrix(int num_rows, int num_cols, std::vector<float> rhs):
    m_num_rows(num_rows),
    m_num_cols(num_cols)
{
    this->allocateMemHost();
    this->allocateMemDevice();

    // Copy vector elements to host shared pointer
    for(int j=0; j < this->m_num_rows; j++){
        for(int i =0; i< this->m_num_cols; i++){
            this->h_mat.get()[j*this->m_num_cols+i] = rhs[j*this->m_num_cols+i];
        }
    }

    this->copyHostToDevice();
}

/**
 * TODO
 */
gpu::Matrix::Matrix(const Matrix& rhs):
    // Since rhs is of type Matrix, we
    // can access its private fields
    m_num_rows(rhs.m_num_rows),
    m_num_cols(rhs.m_num_cols),
    h_mat(rhs.h_mat),
    d_mat(rhs.d_mat)
{}

/**
 * TODO
 */
void gpu::Matrix::allocateMemHost(){

    int size  = this->m_num_cols * this->m_num_rows;
    this->h_mat = std::shared_ptr<float>(new float[size]{0},
                                            [&](float* ptr){ delete[] ptr;});
    
}

/**
 * TODO
 */
void gpu::Matrix::allocateMemDevice(){
    int size  = this->m_num_cols * this->m_num_rows;
    this->d_mat = std::shared_ptr<float>(nullptr,  [&](float* ptr){ cudaFree(ptr);});
    cudaMalloc((void**) &this->d_mat, size*sizeof(float));
}

/**
 * TODO
 */
void gpu::Matrix::copyHostToDevice(){
    int size  = this->m_num_cols * this->m_num_rows;
    cudaMemcpy(this->d_mat.get(), this->h_mat.get(), size*sizeof(float), cudaMemcpyHostToDevice);
}

/**
 * TODO
 */
void gpu::Matrix::copyDeviceToHost(){
    int size  = this->m_num_cols * this->m_num_rows;
    cudaMemcpy(this->h_mat.get(), this->d_mat.get(), size*sizeof(float), cudaMemcpyDeviceToHost);
}


//================================//
// Matrix operations support.
//================================//



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
 * TODO
*/
__global__ void kTranspose(float* W_T, float* W, 
                            int W_num_rows, int W_num_cols){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;

    if(idx < W_num_rows && idy < W_num_cols){
        W_T[idy*W_num_cols + idx] = W[idx*W_num_rows + idy];
    }
}

/**
 * TODO
 * 
 */
void gpu::Matrix::matrixInitializationDevice()
{
    curandGenerator_t gen;
    curandCreateGenerator(&gen, 
                CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 
                1234ULL);

    int size  = this->m_num_cols * this->m_num_rows;
    float mean = 0.0;
    float stddev = 1/sqrtf(1.0/(float)size);

    curandGenerateNormal(gen, this->d_mat.get(), size, mean, stddev);

    curandDestroyGenerator(gen);

}

/**
 * TODO
 */
void gpu::Matrix::deepCopy(gpu::Matrix& rhs){
    this->m_num_rows = rhs.get_num_rows();
    this->m_num_cols = rhs.get_num_cols();

    rhs.copyDeviceToHost();

    for(int j=0; j < this->m_num_rows; j++){
        for(int i=0; i < this->m_num_cols; i++){
           this->h_mat.get()[j*this->m_num_cols+i] = rhs(j,i);
        }
    }

    this->copyHostToDevice();
}

/**
 * TODO 
*/
gpu::Matrix gpu::Matrix::transpose() const{

    gpu::Matrix transpose_mat(this->get_num_cols(), this->get_num_rows());

    int t = 32;
    int bx = (this->get_num_cols() + t - 1)/t;
    int by = (this->get_num_rows() + t - 1)/t;

    dim3 threads(t,t);
    dim3 blocks(bx, by);

    kTranspose<<<blocks, threads>>>(transpose_mat.d_mat.get(), this->d_mat.get(), 
                                    this->get_num_rows(), this->get_num_cols());
    cudaDeviceSynchronize();

    return transpose_mat;
}

/**
 * TODO
 */
void gpu::Matrix::printMat(){
    this->copyDeviceToHost();

    for (int j=0; j< this->m_num_rows; ++j) {
        for (int i=0; i< this->m_num_cols; ++i) {
            std::cout << this->h_mat.get()[j*this->m_num_cols+i] << std::endl;
        }
    } 
}

//================================//
// Operators.
//================================//

/**
 * Overload assignment operator. 
 */
gpu::Matrix& gpu::Matrix::operator=(const Matrix& rhs){
    // Check if object is being assigned to itself.
    if(this == & rhs){
        return *this;
    }

    this->m_num_cols = rhs.get_num_cols();
    this->m_num_rows = rhs.get_num_rows();


    this->h_mat = rhs.h_mat;
    this->d_mat = rhs.d_mat;

    // Return dereferenced pointer to this matrix.
    // Since it will persist after this methode call,
    // dereferencing is safe.
    return *this;
}

/**
 * Overload equality operator.
 * 
 * Two matrices are equal if and only if
 * they have the same dimensions and their
 * corresonding elements are equal.
 * 
 * return true if two matrices are equal,
 *        false otherwise
 */
bool gpu::Matrix::operator==(Matrix& rhs) {

    bool areEqual = true;

    // Variables to store the element of matrices to be compared
    float this_val = 0.0;
    float rhs_val = 0.0;

    // Fixed error for comparison between two given values
    constexpr double epsilon = 0.01; 

    this->copyDeviceToHost();
    rhs.copyDeviceToHost();

    //Check if the dimensions of the two matrices are equal
    if( this->m_num_rows != rhs.get_num_rows() ||
        this->m_num_cols != rhs.get_num_cols()){
            areEqual = false;
    }else{
        // Check if corresponding elements of the two matracies are equal
        for (int j = 0; j < this->m_num_rows; j++){
            for(int i = 0; i < this->m_num_cols; i++){
                this_val = this->h_mat.get()[j*this->m_num_cols+i];
                rhs_val = rhs(j,i);
                if(!(std::abs(this_val - rhs_val) < epsilon)){
                    areEqual = false;
                }
            }
        }
    }

    return areEqual;

}

/**
 * Overload operator[] for read-only operation on elements of this Matrix.
 * Since this matrix is a flatten 2d vector, the index of a given element 
 * can be computed as (row index)*(number of columns of this matrix) + (column index).
 */
const float& gpu::Matrix::operator()(const int& row, const int& col) const{
    return this->h_mat.get()[row*this->m_num_cols + col];
}

/**
 * Overload operator[] for write operation on elements of this Matrix.
 * Since this matrix is a flatten 2d vector, the index of a given element 
 * can be computed as (row index)*(number of columns of this matrix) + (column index).
 */
float& gpu::Matrix::operator()(const int& row, const int& col) {
    return this->h_mat.get()[row*this->m_num_cols + col];
}

/**
 * 
*/
gpu::Vector gpu::Matrix::operator*(const Vector& rhs) const{

    Vector res(this->get_num_cols());

    int threads =32;
    int blocks = (this->get_num_cols() + threads -1)/threads;

    kMatrixVectorMult<<<blocks, threads>>>(res.d_vec.get(), this->d_mat.get(), 
                                            rhs.d_vec.get(), this->get_num_cols());
    cudaDeviceSynchronize();

    return res;
}


//================================//
// Getter methodes.
//================================//

/**
 * Get the number of rows in this Matrix.
 * 
 * @return Number of rows in this Matrix
 */
int gpu::Matrix::get_num_rows() const{
    return m_num_rows;
}

/**
 * Get the number of columns in this Matrix.
 * 
 * @return Number of columns in this Matrix.
 */
int gpu::Matrix::get_num_cols() const{
    return m_num_cols;
}