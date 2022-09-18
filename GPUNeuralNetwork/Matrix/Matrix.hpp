#ifndef GPU_MATRIX
#define GPU_MATRIX

#include <vector>
#include <initializer_list>

namespace gpu{
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
            std::vector<float> m_mat;


        public:
            Matrix(int num_rows, int num_cols);
            Matrix(const Matrix& other);
            Matrix(int num_rows, int num_cols, std::initializer_list<float> ilist);      

            Matrix& operator=(const Matrix& rhs);
            bool operator==(const Matrix& rhs) const;
            const float& operator()(const int& row, const int& col) const;
            float& operator()(const int& row, const int& col);

            void matrixInitialization(); 


            int get_num_rows() const;
            int get_num_cols() const;

    };
}

#endif // End of GPU_MATRIX