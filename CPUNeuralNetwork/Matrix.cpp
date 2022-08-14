#include "Matrix.hpp"
#include "Vector.hpp"
#include <iostream>
#include <algorithm>
#include "random"

/**
 * Constructor for Matrix object with number of rows and columns specified. 
 */
cpu::Matrix::Matrix(int num_rows, 
                    int num_cols):
                    m_num_rows(num_rows),
                    m_num_cols(num_cols),
                    m_mat(num_rows, std::vector<double>(num_cols, 0.0f))
{}

/**
 * Constructor for Matrix object using initializer list
 */
cpu::Matrix::Matrix(std::initializer_list< std::initializer_list<double> > ilist):
    m_mat(ilist.begin(), ilist.end()),
    m_num_rows(ilist.size()),
    m_num_cols(ilist.begin()->size())
{}

/**
 * Copy Constructor. 
 * Invoked when Matrix mat = rhs
 * Since no dynamic memory was allocated, simple copy 
 * member variables from rhs to this matrix. 
 */
cpu::Matrix::Matrix(const Matrix& rhs):
    // Since rhs is of type Matrix, we
    // can access its private fields
    m_num_rows(rhs.m_num_rows),
    m_num_cols(rhs.m_num_cols),
    //Invoke the copy constrcutor for std::vector
    m_mat(rhs.m_mat)
{}

/**
 * Initialize the elements of the matrix to random values that come
 * from a Gaussian Distribtuion centered at 0 with standard deviations of 
 * @f$\sqrt{ \farc{1}{n_{I}}} $ where @f$n_{I}$ is the size of layer @f$I$.
 * 
 */
void cpu::Matrix::matrix_initialization()
{


    std::mt19937 generator;
    double mean = 0.0f;
    double stddev = std::sqrt(1 / static_cast<double>(this->m_num_cols) ); 
    std::normal_distribution<double> normal(mean, stddev);
    for (int j=0; j< this->m_num_rows; ++j) {
        for (int i=0; i< this->m_num_cols; ++i) {
            this->m_mat[j][i] = normal(generator);
        }
    } 

}



/**
 * Overload assignment operator. 
 */
cpu::Matrix& cpu::Matrix::operator=(const Matrix& rhs){
    // Check if object is being assigned to itself.
    if(this == & rhs){
        return *this;
    }

    int new_col_num = rhs.get_num_cols();
    int new_row_num = rhs.get_num_rows();

    // resize this Matrix
    m_mat.resize(new_row_num);
    for(int j=0; j < m_mat.size(); j++){
        m_mat[j].resize(new_col_num);
    }

    // Assign this matrix values elementwise
    for(int j=0; j < new_row_num; j++){
        for(int i=0; i < new_col_num; i++){
            m_mat[j][i] = rhs[j][i];
        }
    }

    // Set the member variables defining the number of rows and columns of this matrix
    m_num_rows = new_row_num;
    m_num_cols = new_col_num;

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
bool cpu::Matrix::operator==(const Matrix& rhs) const{

    bool areEqual = true;

    // Variables to store the element of matrices to be compared
    double this_val = 0.0;
    double rhs_val = 0.0;

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
                this_val = this->m_mat[j][i];
                rhs_val = rhs[j][i];
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
 * The first constant indicates that we are returning a data type that will
 * not be modified. The second const indicates that the methode parameter will not
 * be modified by this methode. The third const indicates that this methode will not
 * modify the memeber variable.
 */
const std::vector<double>& cpu::Matrix::operator[](const int &input) const{
    return m_mat[input];
}

/**
 * Overload operator[] for write operation on elements of this Matrix.
 * Note that Matrix[j][i] = 3 is the same as
 *           auto& temp = Matrix[j]
 *           temp[i] = 3
 */
std::vector<double>& cpu::Matrix::operator[](const int &input) {
    return m_mat[input];
}


/**
 * 
 * Overload subtraction operator without assigment to allow
 * element-wise subtraction to be performed on a Matrix object.
 * 
 */
cpu::Matrix cpu::Matrix::operator-(const cpu::Matrix& rhs) const{
    cpu::Matrix mat(this->m_num_rows, this->m_num_cols);

    for(int j = 0;  j < this->m_num_rows; j++){
        for(int i = 0; i < this->m_num_cols; i++){
            mat[j][i] = this->m_mat[j][i] - rhs[j][i];
        }
    }

    return mat;
}




/**
 * 
 * Overload subtraction operator with assigment to allow
 * element-wise subtraction to be performed to this Matrix.
 *
 */ 
cpu::Matrix& cpu::Matrix::operator-=(const Matrix& rhs){
    for(int j = 0;  j < this->m_num_rows; j++){
        for(int i = 0; i < this->m_num_cols; i++){
            this->m_mat[j][i] -= rhs[j][i];
        }
    }

    return *this;
}

/**
 * 
 * Overload mulitplication operator without assignment
 * in order to allows scalar multiplication
 * to be performed on a Matrix object
 * 
 */
cpu::Matrix cpu::Matrix::operator*(const double& rhs) const{
    cpu::Matrix mat(this->m_num_rows, this->m_num_cols);

    for(int j = 0;  j < this->m_num_rows; j++){
        for(int i = 0; i < this->m_num_cols; i++){
            mat[j][i] = rhs*this->m_mat[j][i];
        }
    }

    return mat;
}

/**
 * 
 * Overload mulitplication operator with assignment
 * in order to allows scalar multiplication
 * to be performed on this Matrix object
 *
 */ 
cpu::Matrix& cpu::Matrix::operator*=(const double& rhs){
    for(int j = 0;  j < this->m_num_rows; j++){
        for(int i = 0; i < this->m_num_cols; i++){
            this->m_mat[j][i] *= rhs;
        }
    }

    return *this;
}

/**
 * 
 * Overload mulitplication operator without assignment
 * in order to allows vector multiplication
 * to be performed on a Matrix object
 * 
 */
cpu::Vector cpu::Matrix::operator*(const cpu::Vector& rhs) const{
    cpu::Vector vec(this->m_num_rows, 0.0f);

    double temp;

    for(int j = 0;  j < this->m_num_rows; j++){
        temp = 0.0f;
        for(int i = 0; i < this->m_num_cols; i++){
            temp += this->m_mat[j][i]*rhs[i];
        }
        vec[j] = temp;
    }

    return vec;
}

/**
 * 
 * Overload mulitplication operator with assignment
 * in order to allows vector multiplication
 * to be performed on this Matrix object
 *
 *
cpu::Matrix& cpu::Matrix::operator*=(const std::vector<double>& rhs){
    for(int j = 0;  j < this->m_num_rows; j++){
        for(int i = 0; i < this->m_num_cols; i++){
            this->m_mat[j][i] *= rhs[i];
        }
    }

    return *this;
}*/

cpu::Matrix cpu::Matrix::transpose() const{
    cpu::Matrix t(this->m_num_rows, this->m_num_cols);

    for(int j = 0;  j < this->m_num_rows; j++){
        for(int i = 0; i < this->m_num_cols; i++){
            t[i][j] = this->m_mat[j][i];
        }
    }

    return t;
}

/**
 * Print the contents of this Matrix.
 */
void cpu::Matrix::printMat(){
    for(int j = 0; j < this->m_num_rows; j++){
        for(int i = 0; i<this->m_num_cols; i++){
            std::cout << this->m_mat[i][j] << std::endl;
        }
    }
}

/**
 * Get the number of rows in this Matrix.
 * 
 * @return Number of rows in this Matrix
 */
int cpu::Matrix::get_num_rows() const{
    return m_num_rows;
}

/**
 * Get the number of columns in this Matrix.
 * 
 * @return Number of columns in this Matrix.
 */
int cpu::Matrix::get_num_cols() const{
    return m_num_cols;
}
