#ifndef GPU_MATRIX
#define GPU_MATRIX

#include <memory>
#include <vector>

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


        public:

            std::shared_ptr<float> h_mat;
            std::shared_ptr<float> d_mat;

            Matrix(int num_rows, int num_cols);
            Matrix(const Matrix& other);
            Matrix(int num_rows, int num_cols, std::vector<float> rhs);      

            void allocateMemHost();
            void allocateMemDevice();
            void copyHostToDevice();
            void copyDeviceToHost();
            
            void matrixInitializationDevice(); 
            void deepCopy(Matrix& rhs);
            Matrix transpose() const;
            void printMat();

            Matrix& operator=(const Matrix& rhs);
            bool operator==(Matrix& rhs);
            const float& operator()(const int& row, const int& col) const;
            float& operator()(const int& row, const int& col);
            Vector operator*(const Vector& rhs) const;
            Matrix operator*(const float& rhs) const;
            Matrix& operator-=(const Matrix& rhs);
            


            int get_num_rows() const;
            int get_num_cols() const;

    };
}

#endif // End of GPU_MATRIX