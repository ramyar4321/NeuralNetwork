#ifndef GPU_VECTOR
#define GPU_VECTOR

#include <memory>
#include <vector>
#include "Scalar.cuh"

namespace gpu{
    // Forward delcare Matrix class to break circular dependancy between
    // Vector and Matrix classes.
    class Matrix;
    class Vector{
        private:
            int m_size;

        public:

            std::shared_ptr<float> h_vec;
            std::shared_ptr<float> d_vec;

            Vector(int size);
            Vector(std::vector<float> rhs);
            Vector( const Vector& other);

            void allocateMemHost();
            void allocateMemDevice();
            void copyHostToDevice();
            void copyDeviceToHost();

            void vectorInitializationDevice();
            gpu::Scalar dot(const gpu::Vector& rhs) const;
            Matrix tensor(const Vector& rhs) const;
            void deepCopy(Vector& rhs);
            //void printVec();

            Vector& operator=(const Vector& rhs);
            //void  operator=(const std::vector<float>& rhs);
            bool operator==(Vector& rhs);
            const float operator[](const int &input) const;
            float& operator[](const int &input);
            Vector operator*(const float& rhs) const;
            Vector& operator*=(const Vector& rhs);
            Vector& operator-=(const Vector& rhs);


            int getSize() const;

    };
}

#endif // End of GPU_VECTOR