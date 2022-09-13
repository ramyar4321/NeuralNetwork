#ifndef CPU_VECTOR
#define CPU_VECTOR

#include <vector>
#include <initializer_list>

namespace cpu{
    // Forward delcare Matrix class to break circular dependancy between
    // Vector and Matrix classes.
    class Matrix;
    class Vector{
        private:
            int m_size;
            std::vector<double> m_vec;

        public:
            Vector(int size, double initial_val);
            Vector(std::initializer_list<double> ilist);

            //void vectorInitialization();
            double dot(const Vector& rhs) const; 
            Matrix tensor(const Vector& rhs) const;

            Vector& operator=(const Vector& rhs);
            bool operator==(const Vector& rhs);
            const double& operator[](const int &input) const;
            double& operator[](const int &input);
            Vector operator*(const double& rhs) const;
            Vector operator*(const Vector& rhs) const;
            Vector& operator*=(const Vector& rhs);
            Vector operator-(const double& rhs) const;
            Vector& operator-=(const Vector& rhs);

            void printVec();

            int getSize() const;

    };
}

#endif // End of CPU_VECTOR