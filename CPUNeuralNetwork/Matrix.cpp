#include "Matrix.hpp"
#include <cassert>
#include <iostream>

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
cpu::Matrix cpu::Matrix::getSubMatrix(int start_ri, int end_ri, int start_ci, int end_ci){

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
 std::vector<double> cpu::Matrix::getCol(int ci, int start_ri, int end_ri){
    assert(start_ri >= 0 && start_ri < m_num_rows);
    assert(end_ri >= 0 && end_ri < m_num_rows);
    assert(ci >= 0 && ci < m_num_cols);


    std::vector<double> col;

    for(int j= start_ri; j <= end_ri; j++){
        col.push_back(m_mat[j][ci]);
    }

    return col;
 }

/**
 * Return row of matrix.
 * 
 * @param ri Index of the row of this matrix to be returned
 * 
 * @return Row of matrix at index ri.
 */
std::vector<double> cpu::Matrix::getRow(int ri){
    std::vector<double> row;

    for(int i; i <= m_num_cols; i++){
        row.push_back(m_mat[ri][i]);
    }

    return row;
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
