#include "ScalarTesting.hpp"
#include "../Matrix/Scalar.cuh"
#include <iostream>

/**
 * Default constructor.
*/
gpu::ScalarTesting::ScalarTesting(){}

/**
 * Test copy constructor
 * 
*/
void gpu::ScalarTesting::testCopyConstructor(){

    gpu::Scalar scalar(1.0f);

    gpu::Scalar copiedScalar = scalar;

    if(copiedScalar == scalar){
        std::cout << "Test passed! Copy constructor of Scalar class produced expected results." << std::endl;
    }else{
        std::cout << "Test failed! Copy constructor of Scalar class produced unexpected results." << std::endl;
    }

}

/**
 * 
 * Test operator= of Scalar class.
 * 
*/
void gpu::ScalarTesting::testEqualOperator(){
    
    gpu::Scalar scalar(1.0f);

    gpu::Scalar copiedScalar(2.0f);

    copiedScalar = scalar;

    if(copiedScalar ==  scalar){
        std::cout << "Test passed! operator= of Scalar class produced expected results." << std::endl;
    } else{
        std::cout << "Test failed! operator= of Scalar class produced unexpected results." << std::endl;
    }

}

/**
 * 
 * Test operator== of Vector class.
 * 
 * Two tests will be conducted.
 * 
 * - The first test will test if operator==
 *    of Scalar class will return false
 *    for two unequal scalars.
 * 
 * - The second test will test if operator==
 *   of Sclar class will return true for
 *   two equal scalar.
 * 
 * 
*/void gpu::ScalarTesting::testIsEqualOperator(){

    bool testPassed = true;

    gpu::Scalar scalar1(1.0f);
    gpu::Scalar scalar2(2.0f);
    gpu::Scalar scalar3(2.0f);

    if(scalar1 == scalar2){
        testPassed = false;
    }

    if(!(scalar2 == scalar3)){
        testPassed = false;
    }

    if(testPassed){
        std::cout << "Test passed! operator== of Scalar class produced expected results." << std::endl;
    }else{
        std::cout << "Test failed! operator== of Scalar class produced unexpected results." << std::endl;
    }
}