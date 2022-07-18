#include "Matrix.hpp"

/**
 * Constructor for Matrix object. 
 */
cpu::Matrix::Matrix(int num_rows, 
                    int num_cols, double& initial_val):
                    m_num_rows(num_rows),
                    m_num_cols(num_cols),
                    m_initial_val(initial_val),
                    m_mat(num_rows, std::vector<double>(num_cols, initial_val))
{}

