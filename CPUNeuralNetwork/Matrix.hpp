#ifndef CPU_MATRIX
#define CPU_MATRIX

#include <vector>

namespace cpu{
    /**
     * Matrix class to store two dimensional mathematical matrix and 
     * to perform needed mathematical operations on such a matrix.
     * The matrix is stored as a vector of vectors using the std::vector.
     * Each matrix object will be of dimensions num_rows by num_cols. 
     * Each element of the matrix will be intialized to initial_val.
     * 
     * This matrix class will only support operations needed in the NeuralNetwork class.
     * The following operations and operators will be supported:
     * - Matrix with Matrix multiplication.
     * - Matrix with Vector multiplcation.
     * - Matrix with Matrix addition.
     * - Matrix with scalar muliplication.
     * - Matrix with scalar subtraction.
     * - Getting a column of a matrix.
     * 
     */
    class Matrix{
        public:
            Matrix(int num_rows, int num_cols, double& initial_val);

        private:
            unsigned int m_num_rows;
            unsigned int m_num_cols;
            double m_initial_val;
            std::vector<std::vector<double> > m_mat;

    };
}

#endif // End of CPU_MATRIX