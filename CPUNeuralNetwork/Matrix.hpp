#ifndef CPU_MATRIX
#define CPU_MATRIX

#include <vector>
#include <initializer_list>

namespace cpu{
    /**
     * Matrix class to store two dimensional mathematical matrix and 
     * to perform needed mathematical operations on such a matrix.
     * The matrix is stored as a vector of vectors using the std::vector.
     * Each matrix object will be of dimensions num_rows by num_cols. 
     * Each element of the matrix will be intialized to 0.0.
     * 
     * This matrix class will only support operations needed in the NeuralNetwork and
     * Dataset classes.
     * The following operations and operators will be supported:
     * - Getting a column of a matrix.
     * - operator=
     * - operator[] 
     * - Getting a sub-Matrix 
     */
    class Matrix{
        private:
            int m_num_rows;
            int m_num_cols;
            std::vector<std::vector<double> > m_mat;


        public:
            Matrix(int num_rows, int num_cols);
            Matrix(const Matrix& other);
            Matrix(std::initializer_list< std::initializer_list<double> > ilist);

            Matrix& operator=(const Matrix& rhs);
            bool operator==(const Matrix& rhs) const;
            const std::vector<double>& operator[](const int &input) const;
            std::vector<double>& operator[](const int &input);

            Matrix getSubMatrix(int& start_ri, int& end_ri, int& start_ci, int& end_ci);
            std::vector<double> getCol(int& ci, int& start_ri, int& end_ri);
            std::vector<double> getCol(int& ci);
            std::vector<double> getRow(int& ri);

            Matrix standardizeMatrix();

            double computeMean(int& ci);
            double computeStd(int& ci);
            double computeStd(int& ci, double &mean);

            int get_num_rows() const;
            int get_num_cols() const;

    };
}

#endif // End of CPU_MATRIX