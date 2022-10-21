#include "CPUNeuralNetwork.hpp"
#include "Testing/NeuralNetTesting.hpp"
#include "Testing/DatasetTesting.hpp"
#include "Dataset.hpp"
#include <iostream>

int main(){
    cpu::NeuralNetwork net(100,100, 50, 0.001);

    cpu::Dataset dat(306, 4,0.75);
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    cpu::Dataset x_train = dat.X_train_split();
    cpu::Dataset x_train_stand = x_train.standardizeDataset();
    std::vector<float> y_train = dat.y_train_split();

    net.fit(x_train_stand, y_train);

    cpu::Dataset x_test = dat.X_test_split();
    cpu::Dataset x_test_stand = x_test.standardizeDataset();
    std::vector<float> y_test = dat.y_test_split();
    dat.setValues(y_test);
    float threeshold = 0.5;
    std::vector<float> y_pred = net.perdict(x_test_stand, threeshold);
    float acc = net.computeAccuracy(y_pred, y_test);
    std::cout << acc << std::endl;
    
    cpu::NeuralNetTesting test_net;
    test_net.test_forwardPropegation();
    test_net.test_backPropegation();
    test_net.test_gradientDescent();


    cpu::DatasetTesting test_dataset;
    test_dataset.test_import_dataset();
    test_dataset.test_X_train_split();
    test_dataset.test_X_test_split();
    test_dataset.test_y_train_split();
    test_dataset.test_y_test_split();

    test_dataset.test_getRow();
    test_dataset.test_getColumn();
    test_dataset.test_standardizeDataset();
    test_dataset.test_setValue();
    

    return 0;
}