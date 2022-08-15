#ifndef CPU_MATRIX
#define CPU_MATRIX

#include <vector>
#include <initializer_list>

namespace cpu{
    // Forward delcare Vector class to break circular dependancy between
    // Vector and Matrix classes.
    class Vector;
    /**
     * Matrix class to store two dimensional mathematical matrix and 
     * to perform needed mathematical operations on such a matrix.
     * The 2D matrix will be stored in a one dimensional vector.
     * The index of each element can be computed as 
     * row_index*number_of_columns + column_index.
     * Each element of the matrix will be intialized to 0.0.
     * 
     * This is not meant to be a general purpose class, rather it will only
     * have methodes that will be used by the Neural Netowrk class.
     */
    class Matrix{
        private:
            int m_num_rows;
            int m_num_cols;
            std::vector<double> m_mat;


        public:
            Matrix(int num_rows, int num_cols);
            Matrix(const Matrix& other);
            Matrix(int num_rows, int num_cols, std::initializer_list<double> ilist);      

            Matrix& operator=(const Matrix& rhs);
            bool operator==(const Matrix& rhs) const;
            const double& operator()(const int& row, const int& col) const;
            double& operator()(const int& row, const int& col);
            Matrix operator-(const Matrix& rhs) const;
            Matrix& operator-=(const Matrix& rhs);
            Matrix operator*(const double& rhs) const;
            Matrix& operator*=(const double& rhs);
            Vector operator*(const Vector& rhs) const;

            void matrix_initialization(); 
            Matrix transpose() const;
            void printMat();


            int get_num_rows() const;
            int get_num_cols() const;

    };
}

#endif // End of CPU_MATRIX