#include "NeuralNetwork.hpp"
#include "Testing/NeuralNetTesting.hpp"
#include "Testing/DatasetTesting.hpp"
#include "Testing/MatrixTesting.hpp"
#include "Testing/VectorTesting.hpp"
#include "Testing/ScalarTesting.hpp"
#include "Specs/specs.cuh"
#include "Dataset.hpp"
#include <iostream>


int main(){

    // Ucomment to run neural network
    /*gpu::NeuralNetwork net(50,50, 10, 0.001);

    gpu::Dataset dat(306, 4,0.75);
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    gpu::Dataset x_train = dat.X_train_split();
    gpu::Dataset x_train_stand = x_train.standardizeDataset();
    std::vector<float> y_train = dat.y_train_split();

    net.fit(x_train_stand, y_train);

    gpu::Dataset x_test = dat.X_test_split();
    gpu::Dataset x_test_stand = x_test.standardizeDataset();
    std::vector<float> y_test = dat.y_test_split();
    dat.setValues(y_test);
    float threeshold = 0.5;
    std::vector<float> y_pred = net.perdict(x_test_stand, threeshold);
    float acc = net.computeAccuracy(y_pred, y_test);
    std::cout << acc << std::endl;*/

    // Uncomment to get GPU specs
    //gpu::getGPUSpecs();
    
    // Uncomment to run tests for the Matrix class.
    /*gpu::MatrixTesting test_mat;
    test_mat.testCopyConstructor();
    test_mat.testVectorConstructor();
    test_mat.testDeepCopy();
    test_mat.testTranspose();
    test_mat.testEqualOperator();
    test_mat.testIsEqualOperator();
    test_mat.testMultOperator();
    test_mat.testSubAssignOperator();*/

    // Uncomment to run tests for the Vector class
    /*gpu::VectorTesting test_vec;
    test_vec.testVectorConstructor();
    test_vec.testCopyConstructor();
    test_vec.testDot();
    test_vec.testTensor();
    test_vec.testDeepCopy();
    test_vec.testEqualOperator();
    test_vec.testMultOperator();
    test_vec.testMultAssignOperator();
    test_vec.testSubAssignOperator();*/

    // Uncomment to run tests for the Scalar class
    /*gpu::ScalarTesting test_scalar;
    test_scalar.testCopyConstructor();
    test_scalar.testEqualOperator();
    test_scalar.testIsEqualOperator();*/

    // Uncomment to run tests for the neural network.
    /*gpu::NeuralNetTesting test_net;
    test_net.test_forwardPropegation();
    test_net.test_backPropegation();
    test_net.test_gradientDescent();*/

    // Uncomment to run tests for Dataset class.
    /*gpu::DatasetTesting test_dataset;
    test_dataset.test_import_dataset();
    test_dataset.test_X_train_split();
    test_dataset.test_X_test_split();
    test_dataset.test_y_train_split();
    test_dataset.test_y_test_split();

    test_dataset.test_getRow();
    test_dataset.test_getColumn();
    test_dataset.test_standardizeDataset();
    test_dataset.test_setValue();*/

    cudaDeviceReset();

    return 0;
}