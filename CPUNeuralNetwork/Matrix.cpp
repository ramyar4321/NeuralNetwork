#include "Matrix.hpp"

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
 * Constructor for Matrix object using std::vector
 */
cpu::Matrix::Matrix(std::initializer_list< std::initializer_list<double> > ilist):
    m_mat(ilist.begin(), ilist.end()),
    m_num_rows(ilist.size()),
    m_num_cols(ilist.begin()->size())
{}


/**
 * Overload assignment operator. 
 */
cpu::Matrix& cpu::Matrix::operator=(const Matrix& rhs){
    // Check if object is being assigned to itself.
    if(this == & rhs){
        return *this;
    }

    int new_col_num = rhs.get_col_num();
    int new_row_num = rhs.get_row_num();

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
 * Get the number of rows in this Matrix.
 * 
 * @return Number of rows in this Matrix
 */
int cpu::Matrix::get_row_num() const{
    return m_num_rows;
}

/**
 * Get the number of columns in this Matrix.
 * 
 * @return Number of columns in this Matrix.
 */
int cpu::Matrix::get_col_num() const{
    return m_num_cols;
}
