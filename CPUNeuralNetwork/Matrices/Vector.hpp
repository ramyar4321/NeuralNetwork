#ifndef CPU_VECTOR
#define CPU_VECTOR

#include <vector>
#include <initializer_list>

namespace cpu{
    // Forward delcare Matrix class to break circular dependancy between
    // Vector and Matrix classes.
    class Matrix;
    /**
     * Vector class to store mathematical vector. 
     * 
     * This is not meant to be a general purpose class, rather it will only
     * have methodes that will be used by the Neural Network class.
    */
    class Vector{
        private:
            int m_size;
            std::vector<float> m_vec;

        public:
            Vector(int size, float initial_val);
            Vector(std::initializer_list<float> ilist);

            void vectorInitialization();
            float dot(const Vector& rhs) const; 
            Matrix tensor(const Vector& rhs) const;

            Vector& operator=(const Vector& rhs);
            bool operator==(const Vector& rhs);
            const float operator[](const int &input) const;
            float& operator[](const int &input);
            Vector operator*(const float& rhs) const;
            Vector& operator*=(const Vector& rhs);
            Vector& operator-=(const Vector& rhs);

            int getSize() const;

    };
}

#endif // End of CPU_VECTOR