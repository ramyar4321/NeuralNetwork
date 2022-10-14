#include "MatrixTesting.hpp"
#include "../Matrix/Matrix.cuh"
#include "../Matrix/Vector.cuh"
#include <vector>
#include <iostream>

gpu::MatrixTesting::MatrixTesting(){}


/**
 * Test copy constructor of Matrix class.
*/
void gpu::MatrixTesting::testCopyConstructor(){
    std::vector<float> mat_ = {1,2,3,4};
    gpu::Matrix mat(2,2, mat_);

    gpu::Matrix copyMat = mat;

    if(copyMat == mat){
        std::cout << "Test passed! Copy constructor produces expected results." << std::endl;
    }else{
        std::cout << "Test failed! Copy constructor produces unexpected results." << std::endl;
    }
}

/**
 * This method tests the constructor of the Matrix 
 * class that uses a vector to construct a matrix.
*/
void gpu::MatrixTesting::testVectorConstructor(){
    bool testPassed = true;

    std::vector<float> mat_ = {1,2,3,4};
    gpu::Matrix mat(2,2, mat_);

    mat.copyDeviceToHost();

    int mat_num_rows = mat.get_num_rows();
    int mat_num_cols = mat.get_num_cols();

    for(int j = 0; j < mat_num_rows; j++){
        for(int i = 0;  i < mat_num_cols; i++){
            if(mat(j,i) != mat_[j*mat_num_cols+i]){
                testPassed = false;
            }
        }
    }

    if(testPassed){
        std::cout << "Test passed! Matrix constructor created matrix using std::vector." << std::endl;
    }else{
        std::cout << "Test failed! Matrix constructor failed to create matrix using std::vector." << std::endl;
    }

}

/**
 * This methode tests the deepCopy Methode of
 * Matrix class. Specifically, we want to determine 
 * if the d_mat is succesfully copied.
 * 
 * In order for this test to succeed three conditions must hold:
 *      1. The dimensions of this matrix must equal the dimensions of 
 *          the copied matrix.
 *      2. The elements of d_mat of this matrix must equal d_mat
 *          of the copied matrix.
 *      3. The address pointed to by d_mat and h_mat of this matrix must 
 *          not changed after copying. 
*/
void gpu::MatrixTesting::testDeepCopy(){

    bool testPassed = true;

    // Matrix to be copied
    std::vector<float> mat = {1,2,3,4};
    gpu::Matrix copiedMatrix(2,2, mat);

    // This matrix
    gpu::Matrix thisMatrix(2,2);
    // Save address pointed to by h_mat and d_mat
    // before copying.
    float* add_hmat_before = thisMatrix.h_mat.get();
    float* add_dmat_before = thisMatrix.d_mat.get();

    // Perform deep copy
    thisMatrix.deepCopy(copiedMatrix);

    // Save address pointed to by h_mat and d_mat
    // after copying.
    float* add_hmat_after = thisMatrix.h_mat.get();
    float* add_dmat_after = thisMatrix.d_mat.get();

    // Tests if dimensions and elements of thisMatrix
    // are of expected values.
    testPassed = (thisMatrix == copiedMatrix);

    // Test if address pointed to by d_mat and h_mat changed
    testPassed = (add_hmat_before == add_hmat_after);
    testPassed = (add_dmat_before == add_dmat_after);
    
    if(testPassed){
        std::cout << "Test passed! deepCopy of Matrix class produced expected results." << std::endl;
    } else{
        std::cout << "Test failed! deppCopy of Matrix class produced unexpected results." << std::endl;
    }
}

/**
 * This methode tests the transpose methode of the matrix class.
*/
void gpu::MatrixTesting::testTranspose(){

    std::vector<float> mat_ = {1,2,3,4};
    gpu::Matrix mat(2,2,mat_);

    // Actual tranpose of matrix 
    gpu::Matrix actual_transpose_mat = mat.transpose();

    // Expected transpose of matrix
    std::vector<float>  expected_transpose_mat_ = {1,3,2,4};
    gpu::Matrix expected_transpose_mat(2,2,expected_transpose_mat_);

    if(actual_transpose_mat == expected_transpose_mat){
        std::cout << "Test passed! Transpose of matrix produced expected results." << std::endl;
    }else{
        std::cout << "Test failed! Transpose of matrix produced unexpected results." << std::endl;
    }

}

/**
 * Test operator= of Matrix class.
 * 
 * Two conditions must hold for test to pass:
 *      1. The dimensions of this matrix must be equal to 
 *          the dimensions of the rhs matrix.
 *      2.  The pointers h_mat and d_mat must point 
 *          to a new memeory location.
*/
void gpu::MatrixTesting::testEqualOperator(){
    
    bool testPassed = true;

    gpu::Matrix rhs_mat(2,2);

    gpu::Matrix this_mat(3,3);

    this_mat = rhs_mat;

    int rhs_num_rows = rhs_mat.get_num_rows();
    int rhs_num_cols = rhs_mat.get_num_cols();

    int this_num_rows = rhs_mat.get_num_rows();
    int this_num_cols = rhs_mat.get_num_cols();

    float* add_rhs_hmat = rhs_mat.h_mat.get();
    float* add_rhs_dmat = rhs_mat.d_mat.get();

    float* add_this_hmat = this_mat.h_mat.get();
    float* add_this_dmat = this_mat.d_mat.get();

    // Check if dimensions of this matrix 
    // and rhs matrix are equal.
    if(this_num_rows != this_num_rows ||
       this_num_cols != this_num_cols){
        testPassed = false;
    }

    // Check if d_mat and h_mat of this matrix
    // point to the memeory location of d_mat and h_mat
    // of rhs matrix
    if(add_this_hmat != add_rhs_hmat ||
       add_this_dmat != add_rhs_dmat){
        testPassed = false;
    }

    if(testPassed){
        std::cout << "Test passed! operator= of Matrix class works as expected." << std::endl;
    } else{
        std::cout << "Test failed! operator= of Matrix class does not work as expected." << std::endl;
    }


}

/**
 * Test operator== of Matrix class.
 * 
 * Two tests will be conducted.
 * 
 * - One test will test if operator== returns
 *   false for two matrices that have same dimensions
 *    but different elements.
 * 
 * - Second test will test if operator== returns
 *   true for two mmatrices that have the same dimensions
 *   and the same corresponding elements.
 * 
 * 
*/
void gpu::MatrixTesting::testIsEqualOperator(){

    bool testPassed = true;

    std::vector<float> mat1_ = {1,2,3,4};
    std::vector<float> mat2_ = {5,6,7,8};

    gpu::Matrix mat1(2,2,mat1_);
    gpu::Matrix mat2(2,2,mat2_);
    gpu::Matrix mat3(2,2,mat2_);

    // Test if operator== returns false
    // for unequal matricies
    if(mat1 == mat2){
        testPassed = false;
    }

    // Test if operator== reutrn true
    // for equal matrices
    if(!(mat2 == mat3)){
        testPassed = false;
    }

    if(testPassed){
        std::cout << "Test passed! operator== produces expected results." << std::endl;
    }else{
        std::cout << "Test failed! operator== produces unexpected results." << std::endl;
    }
}

/**
 * Test both overloads of operator* of
 * the matrix class.
 * 
*/
void gpu::MatrixTesting::testMultOperator(){

    std::vector<float> mat_ = {1,2,3,4,5,6,7,8,9};
    gpu::Matrix mat(3,3,mat_);

    // Test operator* used in Matrix and
    // Vector mutliplication.
    std::vector<float> vec_ = {2,2,2};
    gpu::Vector vec(vec_);

    std::vector<float> expected_vec_ = {12,30,48};
    gpu::Vector expected_vec(expected_vec_);

    gpu::Vector actual_vec = mat*vec; 

    //Test operator* used in Matrix
    // scalar multiplication
    float scalar = 3;
    
    std::vector<float> expected_mat_ = {3, 6, 9, 12, 15, 18, 21, 24, 27};
    gpu::Matrix expected_mat(3,3, expected_mat_);

    gpu::Matrix actual_mat = mat*scalar;

    if(actual_vec == expected_vec || actual_mat == expected_mat){
        std::cout << "Test passed! Both operator* overloads produces expected results." << std::endl;
    }else{
        std::cout << "Test failed! One or both operator* overloads produces unexpected results." << std::endl;
    }
}

/**
 * Test operator-= of Matrix class.
 * 
*/
void gpu::MatrixTesting::testSubAssignOperator(){

    std::vector<float> mat_ = {1,1,1,1};
    gpu::Matrix mat(2,2, mat_);

    std::vector<float> actual_mat_ = {1,2,3,4};
    gpu::Matrix actual_mat(2,2, actual_mat_);

    actual_mat -= mat;

    std::vector<float> expected_mat_ = {0,1,2,3};
    gpu::Matrix expected_mat(2,2, expected_mat_);

    if(actual_mat == expected_mat){
        std::cout << "Test passed! operator-= overload produces expected results." << std::endl;
    }else{
        std::cout << "Test failed! operator-= overload produces unexpected results." << std::endl;
    }
}