#include "VectorTesting.hpp"
#include "../Matrix/Vector.cuh"
#include "../Matrix/Matrix.cuh"
#include <iostream>

gpu::VectorTesting::VectorTesting(){}

/**
 * Test the constructor of Vector class
 * that uses std::vector to construct
 * a Vector object.
*/
void gpu::VectorTesting::testVectorConstructor(){
    
    bool testPassed = true;
    
    std::vector<float> vec_ = {1,2,3};
    gpu::Vector vec(vec_);

    for(int j=0; j < vec_.size(); j++){
        if(vec[j] != vec_[j]){
            testPassed = false;
        }
    }

    if(testPassed){
        std::cout << "Test passed! Vector constructor created vector using std::vector " << std::endl;
    }else{
        std::cout << "Test failed! Vector constructor failed to create vector using std::vector" << std::endl;
    }
}

/**
 * 
 * Test copy constructor of Vector class.
 * 
*/
void gpu::VectorTesting::testCopyConstructor(){

    std::vector<float> vec_ = {1,2,3};
    gpu::Vector vec(vec_);

    gpu::Vector copiedVec = vec;

    if(copiedVec == vec){
        std::cout << "Test passed! Copy constructor of Vector class produced expected results." << std::endl;
    }else{
        std::cout << "Test failed! Copy constructor of Vector class produced unexpected results." << std::endl;
    }

}

/**
 * Test the dot product methode of Vector class.
*/
void gpu::VectorTesting::testDot(){
    
    std::vector<float> vec1_ = {1,2,3};
    gpu::Vector vec1(vec1_);

    std::vector<float> vec2_ = {4,5,6};
    gpu::Vector vec2(vec2_);

    gpu::Scalar actual_res = vec1.dot(vec2);

    float expected_res_ = 32;
    gpu::Scalar expected_res(expected_res_);

    if(actual_res == expected_res){
        std::cout << "Test passed! Dot product of Vector class produced expected results." << std::endl;
    } else{
        std::cout << "Test passed! Dot product of Vector class produced unexpected results." << std::endl;
    }
}

/**
 * Test the tensor product of Vector class.
*/
void gpu::VectorTesting::testTensor(){

    std::vector<float> vec1_ = {1,2};
    gpu::Vector vec1(vec1_);

    std::vector<float> vec2_ = {3,4};
    gpu::Vector vec2(vec2_);

    gpu::Matrix actual_res = vec1.tensor(vec2);

    std::vector<float> expect_res_ = {3, 6, 4, 8};
    gpu::Matrix expect_res(2,2, expect_res_);

    if(actual_res == expect_res){
        std::cout << "Test passed! Tensor product produced expected results" << std::endl;
    }else{
        std::cout << "Test failed! Tensor prduced produced unexpected results" << std::endl;
    }
}

/**
 * Test the deepCopy methode of Vector class.
 * 
 * Specifically, we want to determine 
 * if the d_vec is succesfully copied.
 * 
 * In order for this test to succeed three conditions must hold:
 *      1. The size of this vector must equal the size of the copied vector.
 *      2. The corresponding elements of d_vec of this vector 
 *          must equal the elements of the copied vector.
 *      3. The address pointed to by d_vec and h_vec of this vector must 
 *          not changed after copying. 
*/
void gpu::VectorTesting::testDeepCopy(){

    bool testPassed = true;

    // Vector to be copied
    std::vector<float> vec = {1,2,3,4};
    gpu::Vector copiedVec(vec);

    // This vector
    gpu::Vector thisVec(4);
    // Save address pointed to by h_vec and d_vec
    // before copying.
    float* add_hvec_before = thisVec.h_vec.get();
    float* add_dvec_before = thisVec.d_vec.get();

    // Perform deep copy
    thisVec.deepCopy(copiedVec);

    // Save address pointed to by h_vec and d_vec
    // after copying.
    float* add_hvec_after = thisVec.h_vec.get();
    float* add_dvec_after = thisVec.d_vec.get();

    // Tests if size and elements of thisVec
    // are of expected values.
    testPassed = (thisVec == copiedVec);

    // Test if address pointed to by d_vec and h_vec changed
    testPassed = (add_hvec_before == add_hvec_after);
    testPassed = (add_dvec_before == add_dvec_after);
    
    if(testPassed){
        std::cout << "Test passed! deepCopy of Vector class produced expected results." << std::endl;
    } else{
        std::cout << "Test failed! deppCopy of Vector class produced unexpected results." << std::endl;
    }
}

/**
 * 
 * Test operator= of Vector class.
 * 
 * For this test to pass two conditions must hold true:
 *  1. The size of this vector must be equal to the size
 *      of rhs vector.
 *  2. The address pointed to by h_vec and d_vec of 
 *      this vector must point to the memeory address
 *      pointed by h_vec and d_vec of rhs vector.
 * 
*/
void gpu::VectorTesting::testEqualOperator(){
    bool testPassed = true;

    gpu::Vector rhs(2);
    gpu::Vector thisVec(1);

    thisVec = rhs;

    int rhs_size = rhs.getSize();
    int thisVec_size =  thisVec.getSize();

    float* add_rhs_hvec = rhs.h_vec.get();
    float* add_rhs_dvec = rhs.d_vec.get();

    float* add_thisVec_hvec = thisVec.h_vec.get();
    float* add_thisVec_dvec = thisVec.d_vec.get();

    if(rhs_size != thisVec_size){
        testPassed = false;
    }

    if(add_rhs_hvec != add_thisVec_hvec ||
        add_rhs_dvec != add_thisVec_dvec){
            testPassed = false;
    }

    if(testPassed){
        std::cout << "Test passed! operator= of Vector class produced expected results." << std::endl;
    } else{
        std::cout << "Test failed! operator= of Vector class produced unexpected results." << std::endl;
    }

}

/**
 * Test operator== of Vector class.
 * 
 * Test tests will be conducted.
 * 
 * - One test will test if operator== of Vector class
 *   will return false for two vectors that have the 
 *   same size but different corresponding elements.
 * 
 * - The other test will test if operator== of Vector
 *   class will return true for two vectors that have
 *   the same size and same corresponding elements.
 * 
 * 
*/
void gpu::VectorTesting::testIsEqualOperator(){
    
    bool testPassed = true;

    std::vector<float> vec1_ = {1,2,3};
    std::vector<float> vec2_ = {4,5,6};

    gpu::Vector vec1(vec1_);
    gpu::Vector vec2(vec2_);
    gpu::Vector vec3(vec2_);

    // Test if operator== returns false
    // for vectors that are unequal
    if(vec1 ==  vec2){
        testPassed = false;
    }

    // Test if operator== returns true
    // for vectors that are equal
    if(!(vec2 == vec3)){
        testPassed = false;
    }

    if(testPassed){
        std::cout << "Test passed! operator== of Vector class produced expected results." << std::endl;
    }else{
        std::cout << "Test failed! operator== of Vector class produced unexpected results." << std::endl;
    }

}

/**
 * 
 * Test operator* of Vector class.
 * 
*/
void gpu::VectorTesting::testMultOperator(){

    float scalar = 2;

    std::vector<float> vec_ = {1,2};
    gpu::Vector vec(vec_);

    gpu::Vector actual_res = vec*scalar;

    std::vector<float> expected_res_ = {2,4};
    gpu::Vector expected_res(expected_res_);

    if(actual_res == expected_res){
        std::cout << "Test passed! operator* produced expected results." << std::endl;
    }else{
        std::cout << "Test failed! operator* produced unexpected results." << std::endl;
    }

}


/**
 * Test operator*= of Vector class.
*/
void gpu::VectorTesting::testMultAssignOperator(){

    std::vector<float> vec_ = {1,2};
    gpu::Vector vec(vec_);

    std::vector<float> actual_res_ = {3,4};
    gpu::Vector actual_res(actual_res_);

    actual_res *= vec;

    std::vector<float> expected_res_ = {3,8};
    gpu::Vector expected_res(expected_res_);

    if(actual_res == expected_res){
        std::cout << "Test pass! operator*= of Vector class produced expected results." << std::endl;
    }else{
        std::cout << "Test failed! operator*= of Vector class produced unexpected results." << std::endl;
    }
}

/**
 * 
*/
void gpu::VectorTesting::testSubAssignOperator(){

    std::vector<float> vec_ = {1,2};
    gpu::Vector vec(vec_);

    std::vector<float> actual_res_ = {3,4};
    gpu::Vector actual_res(actual_res_);

    actual_res -= vec;

    std::vector<float> expected_res_ = {2,2};
    gpu::Vector expected_res(expected_res_);

    if(actual_res == expected_res){
        std::cout << "Test pass! operator-= of Vector class produced expected results." << std::endl;
    }else{
        std::cout << "Test failed! operator-= of Vector class produced unexpected results." << std::endl;
    }
}