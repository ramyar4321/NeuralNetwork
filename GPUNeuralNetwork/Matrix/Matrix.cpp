#include "Matrix.hpp"
#include "Vector.hpp"
#include <iostream>
#include <algorithm>
#include "random"

//================================//
// Constructors.
//================================//

/**
 * Constructor for Matrix object with number of rows and columns specified. 
 */
gpu::Matrix::Matrix(int num_rows, 
                    int num_cols):
                    m_num_rows(num_rows),
                    m_num_cols(num_cols)
{
    this->allocateMem();
}

/**
 * Constructor for Matrix object using initializer list
 */
gpu::Matrix::Matrix(int num_rows, int num_cols, std::shared_ptr<float> rhs):
    m_num_rows(num_rows),
    m_num_cols(num_cols),
    m_mat(rhs)
{}

/**
 * Copy Constructor. 
 * Invoked when Matrix mat = rhs
 * Since no dynamic memory was allocated, simple copy 
 * member variables from rhs to this matrix. 
 */
gpu::Matrix::Matrix(const Matrix& rhs):
    // Since rhs is of type Matrix, we
    // can access its private fields
    m_num_rows(rhs.m_num_rows),
    m_num_cols(rhs.m_num_cols),
    //Invoke the copy constrcutor for std::vector
    m_mat(rhs.m_mat)
{}

//================================//
// Matrix operations support.
//================================//



/**
 * Initialize the elements of the matrix to random values that come
 * from a Gaussian Distribtuion centered at 0 with standard deviations of 
 * @f$\sqrt{ \farc{1}{n_{I}}} $ where @f$n_{I}$ is the size of layer @f$I$.
 * 
 */
void gpu::Matrix::matrixInitialization()
{


    std::mt19937 generator;
    float mean = 0.0f;
    float stddev = std::sqrt(1 / static_cast<float>(this->m_num_cols) ); 
    std::normal_distribution<float> normal(mean, stddev);
    for (int j=0; j< this->m_num_rows; ++j) {
        for (int i=0; i< this->m_num_cols; ++i) {
            this->m_mat.get()[j*this->m_num_cols+i] = normal(generator);
        }
    } 

}

void gpu::Matrix::allocateMem(){
    int size = this->m_num_cols* this->m_num_rows;
    this->m_mat = std::shared_ptr<float>(new float[size]{0},
                                        [&](float* ptr){ delete[] ptr; });
}

void gpu::Matrix::deepCopy(const gpu::Matrix& rhs){
    this->m_num_rows = rhs.get_num_rows();
    this->m_num_cols = rhs.get_num_cols();

    for(int j=0; j < this->m_num_rows; j++){
        for(int i=0; i < this->m_num_cols; i++){
           this->m_mat.get()[j*this->m_num_cols+i] = rhs(j,i);
        }
    }
}

void gpu::Matrix::printMat(){
    for (int j=0; j< this->m_num_rows; ++j) {
        for (int i=0; i< this->m_num_cols; ++i) {
            std::cout << this->m_mat.get()[j*this->m_num_cols+i] << std::endl;
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


    this->m_mat = rhs.m_mat;

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
bool gpu::Matrix::operator==(const Matrix& rhs) const{

    bool areEqual = true;

    // Variables to store the element of matrices to be compared
    float this_val = 0.0;
    float rhs_val = 0.0;

    // Fixed error for comparison between two given values
    constexpr double epsilon = 0.01; 

    //Check if the dimensions of the two matrices are equal
    if( this->m_num_rows != rhs.get_num_rows() ||
        this->m_num_cols != rhs.get_num_cols()){
            areEqual = false;
    }else{
        // Check if corresponding elements of the two matracies are equal
        for (int j = 0; j < this->m_num_rows; j++){
            for(int i = 0; i < this->m_num_cols; i++){
                this_val = this->m_mat.get()[j*this->m_num_cols+i];
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
    return this->m_mat.get()[row*this->m_num_cols + col];
}

/**
 * Overload operator[] for write operation on elements of this Matrix.
 * Since this matrix is a flatten 2d vector, the index of a given element 
 * can be computed as (row index)*(number of columns of this matrix) + (column index).
 */
float& gpu::Matrix::operator()(const int& row, const int& col) {
    return this->m_mat.get()[row*this->m_num_cols + col];
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