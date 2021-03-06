#include "CPUTesting.hpp"
#include "CPUNeuralNetwork.hpp"
#include "Dataset.hpp"
#include <vector>
#include <iostream>
#include <functional>
#include "Matrix.hpp"


/*----------------------------------------------*/
// Testing methodes of the NeuralNetowrk class

/**
 * This methode is used to test the compute_outputs method of
 * the NeuralNetwork class. The compute_outputs methode will be
 * tested for each layer of the neural network, thus it 
 * will be tested three times.
 */
void cpu::Testing::test_compute_outputs(){

    // Declare and initialize variables that will be used as
    // input the compute_outputs methode.

    unsigned int input_size = 1;
    unsigned int layer_p_size = 2;
    unsigned int layer_q_size = 2;
    unsigned int layer_r_size = 1;

    // Note, the weights and outputs initialized do not correspond to actual
    // neuron weights and outputs from forward and back propegations. 
    // They are random values simply used to test the compute_output methode.

    cpu::Matrix W1( {{1.1f}, {2.2f}} );
    cpu::Matrix W2 ( {{3.3f, 4.4f}, {5.5f, 6.6f}} );
    cpu::Matrix W3 ( {{7.7f, 8.8f}} ); 


    // Note, a3 does not need to be used since the output neuron 
    // is in the last layer and does not feed into any other neuron.
    std::vector<double> x = {1.1f};
    std::vector<double> a1 = {2.0, 3.0};
    std::vector<double> a2 = {4.0, 5.0};

    // Declare and initialize the expected output of compute_outputs.

    // z1 = { x * W1_11, x * W1_21} = {1.1 * 1.1, 1.1 * 2.2}
    std::vector<double> expected_z1 = {1.21f, 2.42f};
    // z2 = { {a1_1 * W2_11 + a1_2 * W2_12}, {a1_1 * W2_21 + a1_2 * W2_22}}
    // z2 = { {2.0 * 3.3 + 3.0 * 4.4}, {2.0 * 5.5 + 3.0 * 6.6}}
    std::vector<double> expected_z2 = {19.8f, 30.8f}; 
    // z3 = {{a2_1 * W3_11 + a2_2 * W3_12}} = {{4.0* 7.7 + 5.0 * 8.8}}
    std::vector<double> expected_z3 = {74.8f};

    // Instantiate an instance of the Neural Network class
    cpu::NeuralNetwork net(input_size,layer_p_size,layer_q_size,layer_r_size);

    // Use mock inputs to test if methode produces expected results

    std::vector<double> actual_z1 = net.compute_outputs(W1, x, input_size, layer_p_size);
    std::vector<double> actual_z2 = net.compute_outputs(W2, a1, layer_p_size, layer_q_size);
    std::vector<double> actual_z3 = net.compute_outputs(W3, a2, layer_q_size, layer_r_size);

    // Function pointer to helper function to be used as callback function
    // when comparing actual and expected values.
    std::function<bool(double,double)> f = &cpu::Testing::areFloatEqual;

    if ( std::equal(actual_z1.begin(), actual_z1.end(), expected_z1.begin(),f))
        std::cout << "Test succeded for z1.\n";
    else
        std::cout << "Test failed for z1.\n";

    if ( std::equal(actual_z2.begin(), actual_z2.end(), expected_z2.begin(),f))
        std::cout << "Test succeded for z2.\n";
    else
        std::cout << "Test failed for z2.\n";

    if ( std::equal(actual_z3.begin(), actual_z3.end(), expected_z3.begin(),f))
        std::cout << "Test succeded for z3.\n";
    else
        std::cout << "Test failed for z3.\n";


}

/**
 * This methode is used to test the relu_activation method of
 * the NeuralNetwork class. 
 */
void cpu::Testing::test_relu_activation(){

    // Declare and initialize variables that will be used as
    // input the relu_activation methode.

    unsigned int input_size = 1;
    unsigned int layer_p_size = 2;
    unsigned int layer_q_size = 2;
    unsigned int layer_r_size = 1;

    std::vector<double> z1 = {2.28f, 4.19f};
    std::vector<double> z2 = {-2.28f, -4.19f};

    // Declare and initialize the expected output of relu_activation.

    // a1 = { max(0, z1_1), max(0, z1_2)} = { max(0, 2.28f), max(0, 4.19f)}
    std::vector<double> expected_a1 = {2.28f, 4.19f};
    // a2 = { max(0, z2_1), max(0, z2_2)} = { max(0, -2.28f), max(0, -4.19f)}
    std::vector<double> expected_a2 = {0.0f, 0.0f}; 

    // Instantiate an instance of the Neural Network class
    cpu::NeuralNetwork net(input_size,layer_p_size,layer_q_size,layer_r_size);

    // Use mock inputs to test if methode produces expected results
    std::vector<double> actual_a1 = net.relu_activation(z1, layer_p_size);
    std::vector<double> actual_a2 = net.relu_activation(z2, layer_q_size);

    // Function pointer to helper function to be used as callback function
    // when comparing actual and expected values.
    std::function<bool(double,double)> f = &cpu::Testing::areFloatEqual;

     if ( std::equal(actual_a1.begin(), actual_a1.end(), expected_a1.begin(), f))
        std::cout << "Test succeded for a1.\n";
    else
        std::cout << "Test failed for a1.\n";

    if ( std::equal(actual_a2.begin(), actual_a2.end(), expected_a2.begin(),f))
        std::cout << "Test succeded for a2.\n";
    else
        std::cout << "Test failed for a2.\n";


}


/**
 * This methode is used to test the sigmoid_activation method of
 * the NeuralNetwork class. Three tests will be conducted. The sigmoid 
 * method will be tested with positive, negative, and zero values.
 */
void cpu::Testing::test_sigmoid_activation(){

    // Declare and initialize variables that will be used as
    // input the sigmoid_activation methode.

    unsigned int input_size = 1;
    unsigned int layer_p_size = 2;
    unsigned int layer_q_size = 2;
    unsigned int layer_r_size = 1;

    double z_1 = 3.3;
    double z_2 = -3.3;
    double z_3 = 0.0;

    // Declare and initialize the expected output of sigmoid_activation.

    // a = { 1/ (e^(-3.33)) ,  1/ (e^(+3.33)),  1/ (e^(0))} 
    // a = 1/ (e^(-3.33))
    double expected_a_1 = 0.965;
    // a = 1/ (e^(-3.33))
    double expected_a_2 = 0.03456;
    // a = 1/ (e^(0))
    double expected_a_3 = 0.5;

    // Instantiate an instance of the Neural Network class
    cpu::NeuralNetwork net(input_size,layer_p_size,layer_q_size,layer_r_size);

    // Use mock inputs to test if methode produces expected results
    double actual_a_1 = net.sigmoid_activation(z_1);
    double actual_a_2 = net.sigmoid_activation(z_2);
    double actual_a_3 = net.sigmoid_activation(z_3);

    // Function pointer to helper function to be used as callback function
    // when comparing actual and expected values.
    std::function<bool(float,float)> f = &cpu::Testing::areFloatEqual;

    if ( areFloatEqual(expected_a_1, actual_a_1))
        std::cout << "Test succeded for a1.\n";
    else
        std::cout << "Test failed for a1.\n";

    if ( areFloatEqual(expected_a_2, actual_a_2))
        std::cout << "Test succeded for a2.\n";
    else
        std::cout << "Test failed for a2.\n";

    if ( areFloatEqual(expected_a_3, actual_a_3))
        std::cout << "Test succeded for a3.\n";
    else
        std::cout << "Test failed for a3.\n";

}

/**
 * This methode is used to test the compute_loss method of
 * the NeuralNetwork class. We want to test if logarithm of 
 * the entropy loss function is defined for when a is zero.
 * Two tests will be conducted. 
 * 1. y = 1 and a = 0 where y is the actual outcome for a given sample of the dataset
 *    and a is the output of the output neuron. This test will determine if the logarithm
 *    of the first term of the entropy loss function is undefined for the choose output a.
 * 2. y = 0 and a = 1 where y is the actual outcome for a given sample of the dataset
 *    and a is the output of the output neuron. This test will determine if the logarithm
 *    of the second term of the entropy loss function is undefined for the choose output a.
 */
void cpu::Testing::test_compute_loss(){

    // Declare and initialize variables that will be used as
    // input the sigmoid_activation methode.

    unsigned int input_size = 1;
    unsigned int layer_p_size = 2;
    unsigned int layer_q_size = 2;
    unsigned int layer_r_size = 1;

    double y_1 = 1.0;
    double a_1 = 0.0;
    double y_2 = 0.0;
    double a_2 = 1.0;

    // Declare and initialize the expected output of sigmoid_activation.

    // loss = -y*log(a + epsilon) - (1-y)*log(1 - a + epsilon)
    // loss = -log(0.0001) 4
    float expected_loss = 9.21; // Expected loss for both tests.

    // Instantiate an instance of the Neural Network class
    cpu::NeuralNetwork net(input_size,layer_p_size,layer_q_size,layer_r_size);

    // Use mock inputs to test if methode produces expected results
    float actual_loss_1 = net.compute_loss(y_1, a_1);
    float actual_loss_2 = net.compute_loss(y_2, a_2);

    if ( areFloatEqual(expected_loss, actual_loss_1))
        std::cout << "First test succeeded for the entropy loss methode.\n";
    else
        std::cout << "Second test failed for the entropy loss methode.\n";
    if ( areFloatEqual(expected_loss, actual_loss_2))
        std::cout << "First test succeeded for the entropy loss methode.\n";
    else
        std::cout << "SecondTest failed for the entropy loss methode.\n";


}

/*----------------------------------------------*/
// Testing methodes for Dataset class methodes

/**
 * The methode tests the import_dataset
 * methode of the Dataset class. Testing to determine if 
 * every single element of the data file was successfully stored
 * is not feasible. Only the values of the first and last row
 * of the dataset will be test to determine if they were 
 * successfully stored.
 *
 */
void cpu::Testing::test_import_dataset(){

    // Instantiate Dataset object and call methode to be tested.
    cpu::Dataset dat(4, 306,0.75);
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    Matrix dataset = dat.get_dataet();

    // The actual values
    double actual_value1 = dataset[0][0];
    double actual_value2 = dataset[0][1];
    double actual_value3 = dataset[0][2];
    double actual_value4 = dataset[0][3];

    double actual_value5 = dataset[305][0];
    double actual_value6 = dataset[305][1];
    double actual_value7 = dataset[305][2];
    double actual_value8 = dataset[305][3];

    // The expected values
    double expected_value1 = 30;
    double expected_value2 = 64;
    double expected_value3 = 1;
    double expected_value4 = 1;

    double expected_value5 = 83;
    double expected_value6 = 58;
    double expected_value7 = 2;
    double expected_value8 = 2;

    // Test if expected values are the same as the actual values
    if ( areFloatEqual(actual_value1, expected_value1) &&
         areFloatEqual(actual_value2, expected_value2) &&
         areFloatEqual(actual_value3, expected_value3) &&
         areFloatEqual(actual_value4, expected_value4))
        std::cout << "The first four data values of the CSV file successfully imported.\n";
    else
        std::cout << "The first four data values of the CSV file failed to imported.\n";
    if ( areFloatEqual(actual_value5, expected_value5) &&
         areFloatEqual(actual_value6, expected_value6) &&
         areFloatEqual(actual_value7, expected_value7) &&
         areFloatEqual(actual_value8, expected_value8))
        std::cout << "The last four data values of the CSV file successfully imported.\n";
    else
        std::cout << "The last four data values of the CSV file failed to imported.\n";
}

/**
 *
 * This methode tests the X_train_split of the Dataset class.
 * 
 */
void cpu::Testing::test_X_train_split(){

    // Instantiate objects and initialize variables

    // Instantiate Dataset object
    // Train test ratio will be set to 0.99.
    cpu::Dataset dat(4, 306,0.99);
    //Import the dataset
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    Matrix dataset = dat.get_dataet();

    // Call methode to be tested.
    Matrix train_dataset = dat.X_train_split();

    // Actual size of the X train set data
    unsigned int actual_col_size = train_dataset.get_col_num();
    unsigned int actual_row_size = train_dataset.get_row_num();
 
    // Expected size of the X train set data.
    // The expected number of columns is 3 since
    // the training set includes all the columns of the dataset except 
    // the outcome column.
    unsigned int expected_col_size = 3;
    // The expected number of rows is 
    // floor((train test ratio)*(number of rows of the dataset)) = floor(0.99*306) = 302 
    unsigned int expected_row_size = 302;

    // The actual values of the first row of the X train dataset
    double actual_value1 = train_dataset[0][0];
    double actual_value2 = train_dataset[0][1];
    double actual_value3 = train_dataset[0][2];

    // The actual values of the last row of the X train dataset
    double actual_value4 = train_dataset[301][0];
    double actual_value5 = train_dataset[301][1];
    double actual_value6 = train_dataset[301][2];

    // The expected values for the first row of the X train dataset
    // are the values for the first row of the dataset.
    double expected_value1 = 30;
    double expected_value2 = 64;
    double expected_value3 = 1;

    // The expected values for the last row of the X train dataset
    // are the values for the train_sizeth row of the dataset.
    double expected_value4 = 75;
    double expected_value5 = 62;
    double expected_value6 = 1;

    // Conduct tests

    // Test if train dataset is of expected size
    if (actual_col_size == expected_col_size){
        std::cout << "First test succeeded! X training set number has expected number of columns.\n";
    } else{
        std::cout << "First test failed! X training set number has unexpected number of columns.\n";
    }

    if (actual_row_size == expected_row_size){
        std::cout << "Second test succeeded! X training set number has expected number of rows.\n";
    } else{
        std::cout << "Second test failed! X training set number has unexpected number of rows.\n";
    }

    // Test if train dataset has expected values
    if ( areFloatEqual(actual_value1, expected_value1) &&
         areFloatEqual(actual_value2, expected_value2) &&
         areFloatEqual(actual_value3, expected_value3) )
        std::cout << "Third test succeeded! The first row of the X train dataset has expected values.\n ";
    else
        std::cout << "Third test failed! The first row of the X train dataset has unexpected values.\n ";
    if ( areFloatEqual(actual_value4, expected_value4) &&
         areFloatEqual(actual_value5, expected_value5) &&
         areFloatEqual(actual_value6, expected_value6))
        std::cout << "Fourth Test succeeded! The last row of the X train dataset has expected value.\n";
    else
        std::cout << "Fourth test failed! The last row of the X train dataset has unexpected values.\n";

}

/**
 *
 * This methode tests the X_test_split of the Dataset class.
 * 
 */
void cpu::Testing::test_X_test_split(){

    // Instantiate objects and initialize variables

    // Instantiate Dataset object
    // Train test ratio will be set to 0.99.
    cpu::Dataset dat(4, 306,0.99);
    //Import the dataset
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    Matrix dataset = dat.get_dataet();

    // Call methode to be tested.
    Matrix test_dataset = dat.X_test_split();

    // Actual size of the X test set data
    unsigned int actual_col_size = test_dataset.get_col_num();
    unsigned int actual_row_size = test_dataset.get_row_num();
 
    // Expected size of the X test set data.
    // The expected number of columns is 3 since
    // the test set includes all the columns of the dataset except 
    // the outcome column.
    unsigned int expected_col_size = 3;
    // The expected number of rows is 
    // ceiling((1- train test ratio)*(number of rows of the dataset)) = ceiling(0.01*306) = 4
    unsigned int expected_row_size = 4;

    // The actual values of the first row of the X test dataset
    double actual_value1 = test_dataset[0][0];
    double actual_value2 = test_dataset[0][1];
    double actual_value3 = test_dataset[0][2];

    // The actual values of the last row of the X test dataset
    double actual_value4 = test_dataset[3][0];
    double actual_value5 = test_dataset[3][1];
    double actual_value6 = test_dataset[3][2];

    // The expected values for the first row of the X test dataset
    // are the values for the train_sizeth + 1 row of the dataset.
    double expected_value1 = 76;
    double expected_value2 = 67;
    double expected_value3 = 0;

    // The expected values for the last row of the X test dataset
    // are the values for the last row of the dataset.
    double expected_value4 = 83;
    double expected_value5 = 58;
    double expected_value6 = 2;

    // Conduct tests

    // Test if test data set is of expected size
    if (actual_col_size == expected_col_size){
        std::cout << "First test succeeded! X testing set number has expected number of columns.\n";
    } else{
        std::cout << "First test failed! X testing set number has unexpected number of columns.\n";
    }

    if (actual_row_size == expected_row_size){
        std::cout << "Second test succeeded! X testing set number has expected number of rows.\n";
    } else{
        std::cout << "Second test failed! X testing set number has unexpected number of rows.\n";
    }

    // Test if test data set has expected values
    if ( areFloatEqual(actual_value1, expected_value1) &&
         areFloatEqual(actual_value2, expected_value2) &&
         areFloatEqual(actual_value3, expected_value3) )
        std::cout << "Third test succeeded! The first row of the X test dataset has expected values.\n ";
    else
        std::cout << "Third test failed! The first row of the X test dataset has unexpected values.\n ";
    if ( areFloatEqual(actual_value4, expected_value4) &&
         areFloatEqual(actual_value5, expected_value5) &&
         areFloatEqual(actual_value6, expected_value6))
        std::cout << "Fourth Test succeeded! The last row of the X test dataset has expected value.\n";
    else
        std::cout << "Fourth test failed! The last row of the X test dataset has unexpected values.\n";

}

/**
 *
 * This methode tests the y_train_split of the Dataset class.
 * 
 */
void cpu::Testing::test_y_train_split(){

    // Instantiate objects and initialize variables

    // Instantiate Dataset object
    // Train test ratio will be set to 0.99.
    cpu::Dataset dat(4, 306,0.99);
    //Import the dataset
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    Matrix dataset = dat.get_dataet();

    // Call methode to be tested.
    std::vector<double> ytrain_dataset = dat.y_train_split();

    // Actual size of the y train vector
    unsigned int actual_size = ytrain_dataset.size();
 
    // The expected size of y train vector.
    // floor((train test ratio)*(number of rows of the dataset)) = floor(0.99*306) = 302
    unsigned int expected_size = 302;

    // The first actual value of y train vector.
    double actual_value1 = ytrain_dataset[0];

    // The last actual value of the y train vector.
    double actual_value2 = ytrain_dataset[301];

    // The expected value for the first element of y train vector
    // is the value of the first element of the outcome from the dataset.
    double expected_value1 = 1;
    
    // The expected value for the last element of y train vector
    // is the value of the train_sizeth element of the outcome from the dataset.
    double expected_value2 = 1;


    // Conduct tests

    // Test if y train vector is of expected size
    if (actual_size == expected_size){
        std::cout << "First test succeeded! y training vector is of expected size.\n";
    } else{
        std::cout << "First test failed! y train vector is not of expected size.\n";
    }

    // Test if y train vector has expected values
    if ( areFloatEqual(actual_value1, expected_value1) )
        std::cout << "Second test succeeded! The first element of y train vector is of expected value.\n ";
    else
        std::cout << "Second test failed! The first element of y train vector is of unexpected value.\n ";
    if ( areFloatEqual(actual_value2, expected_value2))
        std::cout << "Third Test succeeded! The last element of y train vector is of expected value.\n";
    else
        std::cout << "Third test failed! The last element of y train vector is of unexpected value.\n";

}

/**
 *
 * This methode tests the y_test_split of the Dataset class.
 * 
 */
void cpu::Testing::test_y_test_split(){

    // Instantiate objects and initialize variables

    // Instantiate Dataset object
    // Train test ratio will be set to 0.99.
    cpu::Dataset dat(4, 306,0.99);
    //Import the dataset
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    Matrix dataset = dat.get_dataet();

    // Call methode to be tested.
    std::vector<double> ytest_dataset = dat.y_test_split();

    // Actual size of the y test vector
    unsigned int actual_size = ytest_dataset.size();
 
    // The expected size of y test vector.
    // ceiling((1 - train test ratio)*(number of rows of the dataset)) = floor(0.01*306) = 4
    unsigned int expected_size = 4;

    // The first actual value of y test vector.
    double actual_value1 = ytest_dataset[0];

    // The last actual value of the y test vector.
    double actual_value2 = ytest_dataset[3];

    // The expected value for the first element of y test vector
    // is the value of the train_size+1 element of the outcome from the dataset.
    double expected_value1 = 1;
    
    // The expected value for the last element of y test vector
    // is the value of the last element of the outcome from the dataset.
    double expected_value2 = 2;


    // Conduct tests

    // Test if y train vector is of expected size
    if (actual_size == expected_size){
        std::cout << "First test succeeded! y test vector is of expected size.\n";
    } else{
        std::cout << "First test failed! y test vector is not of expected size.\n";
    }

    // Test if y train vector has expected values
    if ( areFloatEqual(actual_value1, expected_value1) )
        std::cout << "Second test succeeded! The first element of y test vector is of expected value.\n ";
    else
        std::cout << "Second test failed! The first element of y test vector is of unexpected value.\n ";
    if ( areFloatEqual(actual_value2, expected_value2))
        std::cout << "Third Test succeeded! The last element of y test vector is of expected value.\n";
    else
        std::cout << "Third test failed! The last element of y test vector is of unexpected value.\n";

}


/*----------------------------------------------*/
// Helper methodes.

/**
 * Determine if two float values are equal with a fixed error. 
 * Fixed point errors are not used for comparison between floating point values
 * but it will suffice for our usage. 
 */
bool cpu::Testing::areFloatEqual(double a, double b){
    constexpr double epsilon = 0.01; 
    return std::abs(a - b) < epsilon;
}