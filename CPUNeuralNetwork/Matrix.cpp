#include "Matrix.hpp"
#include <cassert>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <math.h>

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
 * not be modified. The second const idicates that the methode parameter will not
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
 * Overload addition operator without assigment to allow
 * scalar element-wise addition to be performed on a Matrix object.
 * 
 */
cpu::Matrix cpu::Matrix::operator+(const double& rhs){
    cpu::Matrix mat(this->m_num_rows, this->m_num_cols);

    for(int j = 0;  j < this->m_num_rows; j++){
        for(int i = 0; i < this->m_num_cols; i++){
            mat[j][i] = this->m_mat[j][i] + rhs;
        }
    }

    return mat;
}

/**
 * 
 * Overload subtraction operator without assigment to allow
 * scalar element-wise subtraction to be performed on a Matrix object.
 * 
 */
cpu::Matrix cpu::Matrix::operator-(const double& rhs){
    cpu::Matrix mat(this->m_num_rows, this->m_num_cols);

    for(int j = 0;  j < this->m_num_rows; j++){
        for(int i = 0; i < this->m_num_cols; i++){
            mat[j][i] = this->m_mat[j][i] - rhs;
        }
    }

    return mat;
}

/**
 * 
 * Overload addition operator with assigment to allow
 * scalar element-wise addition to be performed to this Matrix.
 * 
 */
cpu::Matrix& cpu::Matrix::operator+=(const double& rhs){
    for(int j = 0;  j < this->m_num_rows; j++){
        for(int i = 0; i < this->m_num_cols; i++){
            this->m_mat[j][i] += rhs;
        }
    }

    return *this;
}

/**
 * 
 * Overload subtraction operator with assigment to allow
 * scalar element-wise subtraction to be performed to this Matrix.
 * 
 */
cpu::Matrix& cpu::Matrix::operator-=(const double& rhs){
    for(int j = 0;  j < this->m_num_rows; j++){
        for(int i = 0; i < this->m_num_cols; i++){
            this->m_mat[j][i] -= rhs;
        }
    }

    return *this;
}

/**
 * This methode will produce a submatrix, a block of entries from the original matrix.
 * 
 * @param start_ri The index of the first row of the sub-matrix
 *                 0 <= start_ri < (number of rows in orginal matrix)
 * @param end_ri   The index of the last row of the sub-matrix
 *                 0 <= end_ri < (number of rows in orginal matrix)
 * @param start_ci The index of the first column of the sub-matrix
 *                 0 <= start_ci  < (number of columns in the original matrix)
 * @param end_ci   The index of the last column of the sub-matrix
 *                 0 <= end_ci  < (number of columns in the original matrix)
 * 
 * @return A sub-matrix containing a block of entries of the original matrix.
 * 
 */
cpu::Matrix cpu::Matrix::getSubMatrix(int& start_ri, int& end_ri, int& start_ci, int& end_ci){

    // Assert that Matrix indices are withing the dimensions of this Matrix
    assert(start_ri >= 0 && start_ri < m_num_rows);
    assert(end_ri >= 0 && end_ri < m_num_rows);
    assert(start_ci >= 0 && start_ci < m_num_cols);
    assert(start_ci >= 0 && start_ci < m_num_cols);

    // Calculate dimensions of sub-matrix
    int submat_num_rows = end_ri - start_ri + 1;
    int submat_num_cols = end_ci - start_ci + 1;

    // Create sub-matrix object to be returned
    Matrix submat(submat_num_rows, submat_num_cols);

    for(int j=0, row = start_ri; row <= end_ri; row++, j++){
        for(int i=0, col = start_ci; col <= end_ci; col++, i++){
            submat[j][i] = m_mat[row][col];
        }
    }

    return submat;


}

/**
 * This methode will return all elements from row index start_ri
 * until row index end_ri for the column at index ci. 
 * @param ci       Column index of this matrix corresponding
 * @param start_ri Row index of this matrix corresponding to the first element 
 *                 of the column to be returned.
 * @param end_ri   Row index of this matrix corresponding to the last element 
 *                 of the column to be returned. 
 * @return If start_ri is zero and end_ri is equal to the number of rows in this matrix, 
 *         then this methode will return the column of the matrix at index ci.
 *         Otherwise, it will return a continous segment of the column at index ci. 
 */ 
 std::vector<double> cpu::Matrix::getCol(int& ci, int& start_ri, int& end_ri){
    assert(start_ri >= 0 && start_ri < m_num_rows);
    assert(end_ri >= 0 && end_ri < m_num_rows);
    assert(ci >= 0 && ci < m_num_cols);

    int col_size = end_ri -start_ri +1;


    std::vector<double> col(col_size);

    for(int j= 0, row_i = start_ri; row_i <= end_ri; j++, row_i++){
        col[j] = m_mat[row_i][ci];
    }

    return col;
 }

 /**
  * Return column of matrix
  * 
  * @param ci Index of the column of this matrix to be returned
  * 
  * @return Column of matrix
  * 
  */
std::vector<double> cpu::Matrix::getCol(int& ci){
    std::vector<double> col(m_num_rows);

    for(int j = 0 ; j < m_num_rows; j++){
        col[j] = m_mat[j][ci];
    }

    return col;
}


/**
 * Return row of matrix.
 * 
 * @param ri Index of the row of this matrix to be returned
 * 
 * @return Row of matrix.
 */
std::vector<double> cpu::Matrix::getRow(int& ri){
    std::vector<double> row(m_num_cols);
    //row.reserve(m_num_cols);

    for(int i = 0; i < m_num_cols; i++){
        row[i] = m_mat[ri][i];
    }

    return row;
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
 * Compute the mean of values from a given column.
 * 
 * @param ci Column index for the column of interest from this matrix
 * 
 * @return mean computed for the values from the given column
 */
double cpu::Matrix::computeMean(int& ci){
    std::vector<double> col = getCol(ci);

    double sum = std::accumulate(col.begin(), col.end(), 0.0);
    double mean = sum/col.size();

    return mean;
}

/**
 * Compute the sample standard deviation for the values
 * in a given column. Standard deviation will be computed as such
 * @f$std = \sqrt{\frac{\sum_{j=0}^{n_J} (x_j - \overline{x})}{{n_J}-1}}$
 * where @f$n_J$ is the size of the given column, @f$x_j$ is an element in the 
 * given column, and @f$\overline{x}$ is the mean of for the given column.
 * 
 * Note, that computing the Standard deviation using the following formula
 * @f$std = \sqrt{\frac{\sum_{j=0}^{n_J} (x_j)^2}{{n_J}-1}} -\overline{x}^2$
 * is more prone to overflow or underflow, thus it will not be used here.
 * 
 * @param col A given column from a matrix.
 * 
 * @return Standard deviation for the given column
 */
double cpu::Matrix::computeStd(int& ci){
    std::vector<double> col = getCol(ci);

    double mean  = computeMean(ci);

    double accum = 0.0;
    std::for_each(col.begin(), col.end(), [&](const double x) {
    accum += (x - mean) * (x - mean);
    });

    double std = sqrt(accum/(col.size() -1));

    return std;
}

/**
 * Compute the sample standard deviation for the values
 * in a given column. Standard deviation will be computed as such
 * @f$std = \sqrt{\frac{\sum_{j=0}^{n_J} (x_j - \overline{x})}{{n_J}-1}}$
 * where @f$n_J$ is the size of the given column, @f$x_j$ is an element in the 
 * given column, and @f$\overline{x}$ is the mean of for the given column.
 * 
 * Note, that computing the Standard deviation using the following formula
 * @f$std = \sqrt{\frac{\sum_{j=0}^{n_J} (x_j)^2}{{n_J}-1}} -\overline{x}^2$
 * is more prone to overflow or underflow, thus it will not be used here.
 * 
 * @param col AThe column for which we want the standard deviation
 * @param mean The column for which we want the mean
 * 
 * @return Standard deviation for the given column
 */
double cpu::Matrix::computeStd(int& ci, double& mean){
    std::vector<double> col = getCol(ci);

    double accum = 0.0;
    std::for_each(col.begin(), col.end(), [&](const double x) {
    accum += (x - mean) * (x - mean);
    });

    double std = sqrt(accum/(col.size() -1));

    return std;
}

/**
 * Rescale the data to have a mean of 0 and standard deviation of 1.
 * More percisely, compute the z-score for each element of the matrix.
 * 
 * @return A matrix containing the z-score for each element of this matrix
 * 
 */
cpu::Matrix cpu::Matrix::standardizeMatrix(){
    double col_mean = 0.0;
    double col_std= 0.0;

    Matrix stand_mat(this->m_num_rows, this->m_num_cols);

    for(int i=0; i < this->m_num_cols; i++){
        col_mean = this->computeMean(i);
        col_std = this->computeStd(i);
        for(int j=0; j < this->m_num_rows; j++){
            stand_mat[j][i] = (static_cast<double>(m_mat[j][i]) - col_mean)/col_std;
        }
    }

    return stand_mat;

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
