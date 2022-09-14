#include "CPUNeuralNetwork.hpp"
#include "CPUTesting.hpp"
#include "Dataset.hpp"
#include <iostream>

int main(){
    cpu::NeuralNetwork net(100,100, 50, 0.001);

    cpu::Dataset dat(306, 4,0.75);
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    cpu::Dataset x_train = dat.X_train_split();
    cpu::Dataset x_train_stand = x_train.standardizeMatrix();
    std::vector<double> y_train = dat.y_train_split();

    net.fit(x_train_stand, y_train);

    cpu::Dataset x_test = dat.X_test_split();
    cpu::Dataset x_test_stand = x_test.standardizeMatrix();
    std::vector<double> y_test = dat.y_test_split();
    dat.setValues(y_test);
    double threeshold = 0.5;
    std::vector<double> y_pred = net.perdict(x_test_stand, threeshold);
    double acc = net.computeAccuracy(y_pred, y_test);
    std::cout << acc << std::endl;
    
    cpu::Testing test;
    test.test_forwardPropegation();
    test.test_backPropegation();
    test.test_gradientDescent();
    
    test.test_import_dataset();
    test.test_X_train_split();
    test.test_X_test_split();
    test.test_y_train_split();
    test.test_y_test_split();

    test.test_getRow();
    test.test_getColumn();
    test.test_computeMean();
    test.test_computeStd();
    test.test_standardizeMatrix();
    test.test_setValue();
    

    return 0;
}