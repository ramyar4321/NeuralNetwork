#include "CPUNeuralNetwork.hpp"
#include "CPUTesting.hpp"
#include "Dataset.hpp"

int main(){
    cpu::NeuralNetwork net(3,3);

    /*cpu::Dataset dat(4, 306,0.99);
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);

    cpu::Matrix x_train = dat.X_train_split();
    cpu::Matrix x_train_stand = x_train.standardizeMatrix();
    std::vector<double> y_train = dat.y_train_split();

    net.fit(x_train_stand, y_train);*/




    cpu::Testing test;
    test.test_compute_outputs();
    test.test_relu_activation();
    test.test_sigmoid_activation();
    test.test_compute_loss();
    
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

    //cpu::Dataset dat(4, 306, 0.99);
    //dat.X_train_split();
    //dat.X_test_split();
    //dat.y_test_split();
    

    return 0;
}