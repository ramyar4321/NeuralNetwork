#include "Matrix.cuh"
#include "Vector.cuh"
#include <algorithm>
#include <curand.h>

//================================//
// Constructors.
//================================//

/**
 * Constructor for matrix class. 
 * Constructs matrix with given dimensions.
 * 
 * @param num_rows The number of rows of matrix
 * @param num_cols The number of columns of matrix.
 * 
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
 * Constructor for matrix class.
 * Construct matrix using a vector.
 * 
 * @param num_rows The number of rows of matrix
 * @param num_cols The number of columns of matrix.
 * 
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
 * Copy constructor for matrix class.
 */
gpu::Matrix::Matrix(const Matrix& rhs):
    // Since rhs is of type Matrix, we
    // can access its private fields
    m_num_rows(rhs.m_num_rows),
    m_num_cols(rhs.m_num_cols),
    h_mat(rhs.h_mat),
    d_mat(rhs.d_mat)
{}

//================================//
// Memeory management
//================================//

/**
 * 
 * Allocate matrix on host.
 * Initialize all elements of the matrix to zero.
 * 
 */
void gpu::Matrix::allocateMemHost(){

    int size  = this->m_num_cols * this->m_num_rows;
    this->h_mat = std::shared_ptr<float>(new float[size]{0},
                                            [&](float* ptr){ delete[] ptr;});
    
}

/**
 * Allocate memeory space for matrix on device.
 */
void gpu::Matrix::allocateMemDevice(){
    int size  = this->m_num_cols * this->m_num_rows;
    this->d_mat = std::shared_ptr<float>(nullptr,  [&](float* ptr){ cudaFree(ptr);});
    cudaMalloc((void**) &this->d_mat, size*sizeof(float));
}

/**
 * Copy matrix from host to device.
 */
void gpu::Matrix::copyHostToDevice(){
    int size  = this->m_num_cols * this->m_num_rows;
    cudaMemcpy(this->d_mat.get(), this->h_mat.get(), size*sizeof(float), cudaMemcpyHostToDevice);
}

/**
 * Copy matrix from device to host.
 */
void gpu::Matrix::copyDeviceToHost(){
    int size  = this->m_num_cols * this->m_num_rows;
    cudaMemcpy(this->h_mat.get(), this->d_mat.get(), size*sizeof(float), cudaMemcpyDeviceToHost);
}


//================================//
// CUDA kernels.
//================================//

/**
 * CUDA kernel used in multiplying a matrix by a vector.
 * 
 * @param res A vector containing the results of the matrix-vector multiplcation
 * @param mat The matrix used in matrix-vector multiplication
 * @param vec The vector used in matrix-vector multiplication
 * @param mat_num_cols The numver of columns in the matrix. 
 *                      Assumed to be equal to size of vec.
 * 
 */
__global__ void kMatrixVectorMult(float* res, float* mat, float* vec, int mat_num_cols){
    float temp = 0;

    int idx = blockIdx.x*blockDim.x + threadIdx.x;

    if(idx < mat_num_cols)
        for(int i=0; i < mat_num_cols; i++){
            temp += mat[idx*mat_num_cols + i]*vec[i];
        }
    res[idx] = temp;

}

/**
 * CUDA kernel used in transposing a matrix.
 * 
 * @param mat_T A matrix to store the transpose of matrix mat
 * @param mat   The matrix to be transposed.
 * @param mat_num_rows The number of rows of the matrix mat
 * @param mat_num_cols The number of columns of the matrix mat
 * 
*/
__global__ void kTranspose(float* mat_T, float* mat, 
                            int mat_num_rows, int mat_num_cols){

    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;

    if(idx < mat_num_rows && idy < mat_num_cols){
        mat_T[idy*mat_num_cols + idx] = mat[idx*mat_num_rows + idy];
    }
}

/**
 * CUDA kernel used in multiplying a matrix by a scalar value.
 * 
 * @param res The matrix used to store the result of the matrix-scalar multiplication.
 * @param mat The matrix used matrix-scalar multiplication
 * @param scalar The scalar value used in matrix-scalar multiplcation.
 * @param mat_num_rows The number of columns of the matrix
 * @param mat_num_cols The number of rows of the matrix.
 * 
*/
__global__ void kMatrixScalarMult(float* res, float* mat, float scalar,
                                    int mat_num_rows, int mat_num_cols){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;

    if(idx < mat_num_cols && idy < mat_num_rows){
        res[idy*mat_num_cols + idx] = mat[idy*mat_num_cols +idx]*scalar;
    }
}

/**
 * CUDA kernel used in element-wise subtraction of two matrices
 * and storing the result in one of the two matrices.
 * 
 * @param lhsMat Left hand side matrix used in matrix-matrix elementwise subtraction.
 *               The resulting matrix will be stored in mat1.
 * @param rhsMat Right hand side matrix used in matrix-matrix elementwise subtraction.
 * @param mat_num_rows The number of rows of either matrices.
 * @param mat_num_columns  The number of columns of either matrices.
*/
__global__ void kMatrixMatrixSub(float* lhsMat, float* rhsMat, int mat_num_rows, int mat_num_cols){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;

    if(idx < mat_num_cols && idx < mat_num_rows){
        lhsMat[idy*mat_num_cols +idx] -= rhsMat[idy*mat_num_rows +idx];
    }
}

//================================//
// Matrix support operations.
//================================//

/**
 * Initialize the elements of the matrix to random values that come
 * from a Gaussian Distribtuion centered at 0 with standard deviations of 
 * @f$\sqrt{ \farc{1}{n_{I}}} $ where @f$n_{I}$ is the size of layer @f$I$.
 * Initialization occurs on the device.
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
 * Perform deep copy of matrix. 
 * 
 * That is, set the dimensions of this matrix
 * to the dimensions of rhs matrix. Then copy all
 * corresponding elements of rhs matrix to this matrix.
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
 * Transpose matrix.
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

    return *this;
}

/**
 * Overload equality operator.
 * 
 * Two matrices are equal if and only if
 * they have the same dimensions and their
 * corresonding elements are equal.
 * 
 */
bool gpu::Matrix::operator==(Matrix& rhs) {

    bool areEqual = true;

    // Variables to store the element of matrices to be compared
    float this_val = 0.0;
    float rhs_val = 0.0;

    // Fixed error for comparison between two given values
    constexpr double epsilon = 0.01; 

    // Comparsion is done on host
    // since it is easier.
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
 * Overload multiplication operator without assignment
 * in order to allow mutliplication between matrix and vector.
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

/**
 * 
 * Overload multiplication operator without assignment
 * to allow multiplication between matrix and scalar.
 * 
*/
gpu::Matrix gpu::Matrix::operator*(const float& rhs) const{

    gpu::Matrix res(this->get_num_rows(), this->get_num_cols());

    int t = 32;
    int bx = (this->get_num_cols() + t - 1)/t;
    int by = (this->get_num_rows() + t - 1)/t;

    dim3 threads(t,t);
    dim3 blocks(bx, by);

    kMatrixScalarMult<<<blocks, threads>>>(res.d_mat.get(), this->d_mat.get(), rhs,
                                              this->get_num_rows(), this->get_num_cols());
    cudaDeviceSynchronize();

    return res;
}

/**
 * Overload subtraction operator with assignment
 * in order to allow elementwise subtraction between two matrices.
*/
gpu::Matrix& gpu::Matrix::operator-=(const gpu::Matrix& rhs){
    int t = 32;
    int bx = (this->get_num_cols() + t - 1)/t;
    int by = (this->get_num_rows() + t - 1)/t;

    dim3 threads(t,t);
    dim3 blocks(bx, by);

    kMatrixMatrixSub<<<blocks, threads>>>(this->d_mat.get(), rhs.d_mat.get(),
                                              this->get_num_rows(), this->get_num_cols());
    cudaDeviceSynchronize();

    return *this;
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