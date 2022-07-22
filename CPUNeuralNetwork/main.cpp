#include "CPUNeuralNetwork.hpp"
#include "CPUTesting.hpp"
#include "Dataset.hpp"

int main(){
    //cpu::NeuralNetwork net(3,3,3,1);
    //net.fit();

    cpu::Testing test;
    //test.test_compute_outputs();
    //test.test_relu_activation();
    //test.test_sigmoid_activation();
    //test.test_compute_loss();
    
    test.test_import_dataset();
    test.test_X_train_split();
    test.test_X_test_split();
    test.test_y_train_split();
    test.test_y_test_split();


    //cpu::Dataset dat(4, 306, 0.99);
    //dat.X_train_split();
    //dat.X_test_split();
    //dat.y_test_split();
    

    return 0;
}