#ifndef GPU_VECTOR
#define GPU_VECTOR

#include <memory>

namespace gpu{
    // Forward delcare Matrix class to break circular dependancy between
    // Vector and Matrix classes.
    class Matrix;
    class Vector{
        private:
            int m_size;
            std::shared_ptr<float> m_vec;

        public:
            Vector(int size, float initial_val);
            Vector(int size, std::shared_ptr<float> rhs);

            void vectorInitialization();
            void allocateMem(float initial_val);
            void deepCopy(const Vector& rhs);
            void printVec();

            Vector& operator=(const Vector& rhs);
            bool operator==(const Vector& rhs);
            const float operator[](const int &input) const;
            float& operator[](const int &input);


            int getSize() const;

    };
}

#endif // End of GPU_VECTOR