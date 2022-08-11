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
            Matrix operator-(const Matrix& rhs) const;
            Matrix& operator-=(const Matrix& rhs);
            Matrix operator*(const double& rhs) const;
            Matrix& operator*=(const double& rhs);
            std::vector<double> operator*(const std::vector<double>& rhs) const;
            //Matrix& operator*=(const std::vector<double>& rhs);

            Matrix transpose() const;
            void printMat();


            int get_num_rows() const;
            int get_num_cols() const;

    };
}

#endif // End of CPU_MATRIX