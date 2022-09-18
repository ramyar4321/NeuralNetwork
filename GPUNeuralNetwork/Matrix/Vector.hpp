#ifndef GPU_VECTOR
#define GPU_VECTOR

#include <vector>
#include <initializer_list>

namespace gpu{
    // Forward delcare Matrix class to break circular dependancy between
    // Vector and Matrix classes.
    class Matrix;
    class Vector{
        private:
            int m_size;
            std::vector<float> m_vec;

        public:
            Vector(int size, float initial_val);
            Vector(std::initializer_list<float> ilist);

            void vectorInitialization();

            Vector& operator=(const Vector& rhs);
            bool operator==(const Vector& rhs);
            const float operator[](const int &input) const;
            float& operator[](const int &input);


            int getSize() const;

    };
}

#endif // End of GPU_VECTOR