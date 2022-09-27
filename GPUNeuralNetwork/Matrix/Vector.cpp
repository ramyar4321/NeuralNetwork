#include "Vector.hpp"
#include "Matrix.hpp"
#include "random"
#include <iostream>

/**
 * Constructor for Vector object with size of vector and
 * initial values for each element are specified.
 */
gpu::Vector::Vector(int size, float initial_val):
                    m_size(size)
{
    this->allocateMem(initial_val);
}

/**
 * Constructor for Vector object with initializer list.
 */
gpu::Vector::Vector(int size, std::shared_ptr<float> rhs):
                        m_size(size),
                        m_vec(rhs)
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
        this->m_vec.get()[i] = normal(generator);
    }

}

void gpu::Vector::allocateMem(float initial_val){
    this->m_vec = std::shared_ptr<float>(new float[this->m_size]{initial_val},
                                        [&](float* ptr){ delete[] ptr; });
}

void gpu::Vector::deepCopy(const gpu::Vector& rhs){
    this->m_size = rhs.getSize();

    for(int j= 0 ; j < this->m_size; j++){
        this->m_vec.get()[j] = rhs[j];
    }
}

void gpu::Vector::printVec(){
    for (int i=0; i< this->m_size; ++i) {
        std::cout << this->m_vec.get()[i] << std::endl;
    }
}

gpu::Vector& gpu::Vector::operator=(const Vector& rhs){
    // Check if object is being assigned to itself.
    if(this == &rhs){
        return *this;
    }

    this->m_size = rhs.getSize();

    this->m_vec = rhs.m_vec;

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
                this_val = this->m_vec.get()[i];
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
    return m_vec.get()[input];
}

/**
 * Overload operator[] for write operation on elements of this Vector.
 */
float& gpu::Vector::operator[](const int &input) {
    return m_vec.get()[input];
}


/**
 * Return size of this vector.
 */
int gpu::Vector::getSize() const{
    return this->m_size;
}