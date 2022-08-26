#include "NeuralNetwork.hpp"
#include "Dataset.hpp"
#include <iostream>

int main(){

    gpu::Dataset dat(306, 4,0.90);
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    gpu::Dataset x_train = dat.X_train_split();
    gpu::Dataset x_train_stand = x_train.standardizeMatrix();
    std::vector<float> y_train = dat.y_train_split();

    gpu::NeuralNetwork net(3,100, 100, 1, 1, 0.001);
    net.fit(x_train_stand);

    gpu::Dataset x_test = dat.X_test_split();
    gpu::Dataset x_test_stand = x_test.standardizeMatrix();
    std::vector<float> y_test = dat.y_test_split();
    dat.setValues(y_test);
    float threeshold = 0.5;
    std::vector<float> y_pred = net.perdict(x_test_stand, threeshold);
    float acc = net.computeAccuracy(y_pred, y_test);
    std::cout << acc << std::endl;


    return 0;

}