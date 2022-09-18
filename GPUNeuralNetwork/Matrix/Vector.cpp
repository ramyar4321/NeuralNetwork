#include "Vector.hpp"
#include "Matrix.hpp"
#include "random"
#include <iostream>

/**
 * Constructor for Vector object with size of vector and
 * initial values for each element are specified.
 */
gpu::Vector::Vector(int size, float initial_val):
                    m_size(size),
                    m_vec(size, initial_val)
{}

/**
 * Constructor for Vector object with initializer list.
 */
gpu::Vector::Vector(std::initializer_list<float> ilist):
                        m_vec(ilist.begin(), ilist.end()),
                        m_size(ilist.size())
{}

/**
 * Initialize the elements of the mvector to random values that come
 * from a Gaussian Distribtuion centered at 0 with standard deviations of 
 * @f$\sqrt{ \farc{1}{n_{I}}} $ where @f$n_{I}$ is the size of layer @f$I$.
 * 
 */
void gpu::Vector::vectorInitialization()
{

    std::mt19937 generator;
    float mean = 0.0f;
    float stddev = std::sqrt(1 / static_cast<float>(this->m_size) ); 
    std::normal_distribution<float> normal(mean, stddev);
    for (int i=0; i< this->m_size; ++i) {
        this->m_vec[i] = normal(generator);
    }

}


gpu::Vector& gpu::Vector::operator=(const Vector& rhs){
    // Check if object is being assigned to itself.
    if(this == &rhs){
        return *this;
    }

    int new_size = rhs.getSize();

    // Resize this array
    this->m_vec.resize(new_size);

    // Copy values in this vector
    for(int i=0; i < new_size; i++){
        this->m_vec[i] = rhs[i];
    }

    // Set memeber variables
    this->m_size = new_size;

    return *this;

}

/**
 * Overload equality operator.
 * 
 * Two vectora are equal if and only if
 * they have the same size and their
 * corresonding elements are equal.
 * 
 * return true if two vectors are equal,
 *        false otherwise
 */
bool gpu::Vector::operator==(const Vector& rhs){
    bool areEqual = true;

    // Variables to store the element of vectors to be compared
    float this_val = 0.0;
    float rhs_val = 0.0;

    // Fixed error for comparison between two given values
    constexpr double epsilon = 0.01; 

    //Check if the sizes of the two vectors are equal
    if( this->m_size != rhs.getSize()){
            areEqual = false;
    }else{
        // Check if corresponding elements of the two vectors are equal
            for(int i = 0; i < this->m_size; i++){
                this_val = this->m_vec[i];
                rhs_val = rhs[i];
                if(!(std::abs(this_val - rhs_val) < epsilon)){
                    areEqual = false;
                }
            }

    }

    return areEqual;
}

/**
 * Overload operator[] for read-only operation on elements of this Vector.
 */
const float gpu::Vector::operator[](const int &input) const{
    return m_vec[input];
}

/**
 * Overload operator[] for write operation on elements of this Vector.
 */
float& gpu::Vector::operator[](const int &input) {
    return m_vec[input];
}


/**
 * Return size of this vector.
 */
int gpu::Vector::getSize() const{
    return this->m_size;
}