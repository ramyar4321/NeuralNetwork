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


    int layer_p_size = 2;
    int layer_q_size = 2;


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
    cpu::NeuralNetwork net(layer_p_size,layer_q_size);

    // Use mock inputs to test if methode produces expected results

    std::vector<double> actual_z1 = net.compute_outputs(W1, x);
    std::vector<double> actual_z2 = net.compute_outputs(W2, a1);
    std::vector<double> actual_z3 = net.compute_outputs(W3, a2);

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

    int layer_p_size = 2;
    int layer_q_size = 2;

    std::vector<double> z1 = {2.28f, 4.19f};
    std::vector<double> z2 = {-2.28f, -4.19f};

    // Declare and initialize the expected output of relu_activation.

    // a1 = { max(0, z1_1), max(0, z1_2)} = { max(0, 2.28f), max(0, 4.19f)}
    std::vector<double> expected_a1 = {2.28f, 4.19f};
    // a2 = { max(0, z2_1), max(0, z2_2)} = { max(0, -2.28f), max(0, -4.19f)}
    std::vector<double> expected_a2 = {0.0f, 0.0f}; 

    // Instantiate an instance of the Neural Network class
    cpu::NeuralNetwork net(layer_p_size,layer_q_size);

    // Use mock inputs to test if methode produces expected results
    std::vector<double> actual_a1 = net.relu_activation(z1);
    std::vector<double> actual_a2 = net.relu_activation(z2);

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


    int layer_p_size = 2;
    int layer_q_size = 2;

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
    cpu::NeuralNetwork net(layer_p_size,layer_q_size);

    // Use mock inputs to test if methode produces expected results
    double actual_a_1 = net.sigmoid(z_1);
    double actual_a_2 = net.sigmoid(z_2);
    double actual_a_3 = net.sigmoid(z_3);

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

    int layer_p_size = 2;
    int layer_q_size = 2;


    double y_1 = 1.0;
    double a_1 = 0.0;
    double y_2 = 0.0;
    double a_2 = 1.0;

    // Declare and initialize the expected output of sigmoid_activation.

    // loss = -y*log(a + epsilon) - (1-y)*log(1 - a + epsilon)
    // loss = -log(0.0001) 4
    float expected_loss = 9.21; // Expected loss for both tests.

    // Instantiate an instance of the Neural Network class
    cpu::NeuralNetwork net(layer_p_size,layer_q_size);

    // Use mock inputs to test if methode produces expected results
    float actual_loss_1 = net.bceLoss(y_1, a_1);
    float actual_loss_2 = net.bceLoss(y_2, a_2);

    if ( areFloatEqual(expected_loss, actual_loss_1))
        std::cout << "First test succeeded for the entropy loss methode.\n";
    else
        std::cout << "Second test failed for the entropy loss methode.\n";
    if ( areFloatEqual(expected_loss, actual_loss_2))
        std::cout << "First test succeeded for the entropy loss methode.\n";
    else
        std::cout << "SecondTest failed for the entropy loss methode.\n";


}

/**
 * 
 * This methode tests both the computeDeltaInit
 * and computeGradientInit. That is, test if the correct
 * gradients are compute for m_dLdW3.The finite difference 
 * will be used to approximate the expected gradient. 
 * 
 * The algorithm is as follows:
 * 1. Radomly initialize weight of neural network from a Guassian distribution. 
 * 2. Perform forward and backpropegation to determine the gradients computed
 *    for the last layer of the neural network and store the result. 
 * 3. For each gradient w of the last weight:
 *      - compute negative perturbation: w_minus = w- perturb
 *      - perform forward propegation of neural network 
 *        with w_minus instead of w
 *      - compute loss_minus which is the loss of the neural network
 *         by replacing w with w_minus.
 *      - compute positve pertubation: w_positive = w + perturb
 *      - perform forward propegation of neural network with 
 *        w_positive instead of w
 *      - compute loss_positive which is the loss of the neural network
 *        by replacing w with w_positive.
 *      - estimate numerical gradient for w by using the finite difference methode
 *        numericGradient = (loss_positive - loss_negative)/2*perturb
 *      - store numericGradient.
 * 4. Compare the gradients for the last layer produced by backpropegation
 *    with the numerical estimated gradients.
 */
void cpu::Testing::test_backPropegationInit(){

    cpu::NeuralNetwork net(10,10);

    std::vector<double> x = {-2.11764, 0.3571 , -0.423171};
    double y = 0;

    cpu::Matrix W1(10,3);
    cpu::Matrix W2(10,10);
    std::vector<double> W3(10);

    double a3;

    cpu::Matrix W1_minus(10,3);
    cpu::Matrix W1_plus(10,3);

    cpu::Matrix W2_minus(10,10);
    cpu::Matrix W2_plus(10,10);

    std::vector<double> W3_minus(10);
    std::vector<double> W3_plus(10);

    cpu::Matrix numericdLdW1(10,3);
    cpu::Matrix numericdLdW2(10,10);
    std::vector<double> numericdLdW3(10);

    double perturb = 0.0001;

    double loss_minus;
    double loss_plus;

    net.weight_initialization(W1);
    net.weight_initialization(W2);
    net.weight_initialization(W3);

    net.x(x);
    net.W1(W1);
    net.W2(W2);
    net.W3(W3);
    net.y(y);

    net.forward_propegation();
    net.backPropegation();

    const std::vector<double> &actual_dLdW3 = net.dLdW3();
    const cpu::Matrix& actual_dLdW2 = net.dLdW2();
    const cpu::Matrix& actual_dLdW1 = net.dLdW1();

    for(int i=0; i < W3.size(); i++){
        W3_minus = W3;
        W3_plus = W3;
        W3_minus[i] -= perturb;
        W3_plus[i] += perturb;

        net.W3(W3_minus);
        net.forward_propegation();

        a3= net.a3();

        loss_minus = net.bceLoss(y,a3);

        net.W3(W3_plus);
        net.forward_propegation();
        a3 = net.a3();
        loss_plus =net.bceLoss(y, a3);

        numericdLdW3[i] = (loss_plus-loss_minus)/(2*perturb);      
    }

    for (int j = 0; j < W2.get_num_rows(); j++){
        for(int i=0; i < W2.get_num_cols(); i++){
            W2_minus = W2;
            W2_plus = W2;
            W2_minus[j][i] -= perturb;
            W2_plus[j][i] += perturb;

            net.W2(W2_minus);
            net.forward_propegation();

            a3= net.a3();

            loss_minus = net.bceLoss(y,a3);

            net.W2(W2_plus);
            net.forward_propegation();
            a3 = net.a3();
            loss_plus =net.bceLoss(y, a3);

            numericdLdW2[j][i] = (loss_plus-loss_minus)/(2*perturb);
        }
    }

    for (int j = 0; j < W2.get_num_rows(); j++){
        for(int i=0; i < W2.get_num_cols(); i++){
            W1_minus = W1;
            W1_plus = W1;
            W1_minus[j][i] -= perturb;
            W1_plus[j][i] += perturb;

            net.W1(W1_minus);
            net.forward_propegation();

            a3= net.a3();

            loss_minus = net.bceLoss(y,a3);

            net.W1(W1_plus);
            net.forward_propegation();
            a3 = net.a3();
            loss_plus =net.bceLoss(y, a3);

            numericdLdW1[j][i] = (loss_plus-loss_minus)/(2*perturb);
        }
    }

    //numericdLdW2.printMat();
    for(int j = 0; j < numericdLdW2.get_num_rows(); j++){
        for(int i = 0; i<numericdLdW2.get_num_cols(); i++){
            std::cout << actual_dLdW2[j][i] << std::endl;
            std::cout << numericdLdW2[j][i] << std::endl;
        }
    }
    for(int j = 0; j < numericdLdW1.get_num_rows(); j++){
        for(int i = 0; i<numericdLdW1.get_num_cols(); i++){
            std::cout << actual_dLdW1[j][i] << std::endl;
            std::cout << numericdLdW1[j][i] << std::endl;
        }
    }

    std::function<bool(double,double)> f = &cpu::Testing::areFloatEqual;
    if ( std::equal(actual_dLdW3.begin(), actual_dLdW3.end(), numericdLdW3.begin(), f))
        std::cout << "Test succeeded! Backpropegation gradient matches numeric gradient for last layer.\n";
    else
        std::cout << "Test failed! Backpropegation gradient does not match numeric gradient for last layer.\n";
    
    if(actual_dLdW2 == numericdLdW2){
        std::cout << "Test succeeded! Backpropegation gradient matches numeric gradient for second layer.\n";
    } else{
        std::cout << "Test failed! Backpropegation gradient does not match numeric gradient for second layer.\n";
    }

    if(actual_dLdW2 == numericdLdW2){
        std::cout << "Test succeeded! Backpropegation gradient matches numeric gradient for first layer.\n";
    } else{
        std::cout << "Test failed! Backpropegation gradient does not match numeric gradient for first layer.\n";
    }
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
    unsigned int actual_col_size = train_dataset.get_num_cols();
    unsigned int actual_row_size = train_dataset.get_num_rows();
 
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
    unsigned int actual_col_size = test_dataset.get_num_cols();
    unsigned int actual_row_size = test_dataset.get_num_rows();
 
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

/**
 * This methode tests the setValue methode of the Dataset class.
 */
void cpu::Testing::test_setValue(){

    // Instantiated Dataset object.
    // The parameters are not important. We smiply
    // need an object to access the setValue methode.
    cpu::Dataset dat(4, 306,0.99);

    std::vector<double> y_actual = {2,1,2,1};

    std::vector<double> y_expect = {1,0,1,0};

    dat.setValues(y_actual);

    // Function pointer to helper function to be used as callback function
    // when comparing actual and expected values.
    std::function<bool(double,double)> f = &cpu::Testing::areFloatEqual;

    // Test of setValue returned expected results.
    if ( std::equal(y_actual.begin(), y_actual.end(), y_expect.begin(),f))
        std::cout << "Test succeeded! setValue methode returned expected results.\n";
    else
        std::cout << "Test failed! setValue methode returned unexpected results.\n";

}

/**
 * This methode tests the computeMean methode 
 * of the Matrix clas.
 */
void cpu::Testing::test_computeMean(){
    // Matrix used for testing
    Matrix mat = {{1,2,3},
                  {4,5,6},
                  {7,8,9}};

    int ci = 0;

    // Test to see if zeroth column, considering zero indexing, mean was correctly computed.
    double actual_result= mat.computeMean(ci);
    double expected_result = 4;

    // Function pointer to helper function to be used as callback function
    // when comparing actual and expected values.
    std::function<bool(double,double)> f = &cpu::Testing::areFloatEqual;

     if ( areFloatEqual(actual_result, expected_result))
        std::cout << "Test succeeded! computeMean methode returned expected results.\n";
    else
        std::cout << "Test failed! computeMean methode returned unexpected results.\n";

    
}

/**
 * This methode will test both overload computeStd methodes.
 * There will be two tests, one for each overload methode.
 */
void cpu::Testing::test_computeStd(){

    // Matrix used for testing
    Matrix mat = {{1,2,3},
                  {4,5,6},
                  {7,8,9}};

    int ci = 0;

    // Test to determine if both computeStd methodes
    // compute the correct Standard deviation for the zeroth column.
    double actual_result1= mat.computeStd(ci);

    double mean  = mat.computeMean(ci);
    double actual_result2 = mat.computeStd(ci, mean);
    
    double expected_result = 3;

    // Function pointer to helper function to be used as callback function
    // when comparing actual and expected values.
    std::function<bool(double,double)> f = &cpu::Testing::areFloatEqual;

     if ( areFloatEqual(actual_result1, expected_result))
        std::cout << "Test succeeded! computeStd(ci) methode returned expected results.\n";
    else
        std::cout << "Test failed! computeStd(ci) methode returned unexpected results.\n";
        if ( areFloatEqual(actual_result2, expected_result))
        std::cout << "Test succeeded! computeStd(ci, mean) methode returned expected results.\n";
    else
        std::cout << "Test failed! computeStd(ci, mean) methode returned unexpected results.\n";

}

void cpu::Testing::test_standardizeMatrix(){
    
    Matrix mat = {{1,2,3},  // Matrix used for testing
                  {4,5,6},
                  {7,8,9},
                  {10,11,12}};

    Matrix actual_result = mat.standardizeMatrix();

    Matrix expected_result = {{-1.1619,-1.1619,-1.1619},
                              {-0.387298,-0.387298,-0.387298},
                              {0.387298,0.387298,0.387298},
                              {1.1619,1.1619,1.1619}};


    // Test if standardizeMatrix produced expected output.
    if ( actual_result == expected_result)
        std::cout << "Test succeeded! standardizeMatrix methode returned expected results.\n";
    else
        std::cout << "Test failed! standardizeMatrix methode returned unexpected results.\n";


                  
}

/*----------------------------------------------*/
// Test Matrix methodes

/**
 * Test the getRow methode of the Matrix class
 */
void cpu::Testing::test_getRow(){
    // Matrix used for testing
    Matrix mat = {{1,2,3},
                  {4,5,6},
                  {7,8,9}};

    // Test to see if first row, considering zero indexing, was retreived.
    int ri = 1;
    std::vector<double> actual_results = mat.getRow(ri);
    std::vector<double> expected_results = {4,5,6};

    // Function pointer to helper function to be used as callback function
    // when comparing actual and expected values.
    std::function<bool(double,double)> f = &cpu::Testing::areFloatEqual;

    if ( std::equal(actual_results.begin(), actual_results.end(), expected_results.begin(), f))
        std::cout << "Test succeeded! getRow methode returned expected results.\n";
    else
        std::cout << "Test failed! getROw methode returned unexpected results.\n";

}

/**
 * This methode will test both overload getColumn methodes.
 * There will be two tests, one for each overload methode.
 */
void cpu::Testing::test_getColumn(){
    
    //Matrix used for testing
    Matrix mat = {{1,2,3,4},
                  {5,6,7,8},
                  {9,10,11,12},
                  {13,14,15,16}};

    // Test if the first column, considering zero indexing, was retreived
    int ci = 1;

    int start_ri = 0;
    int end_ri = 2;
    std::vector<double> actual_results1 = mat.getCol(ci, start_ri, end_ri);
    std::vector<double> expected_results1 = {2,6,10};

    
    std::vector<double> actual_results2 = mat.getCol(ci);
    std::vector<double> expected_results2 = {2,6,10,14};

    // Function pointer to helper function to be used as callback function
    // when comparing actual and expected values.
    std::function<bool(double,double)> f = &cpu::Testing::areFloatEqual;

     if ( std::equal(actual_results1.begin(), actual_results1.end(), expected_results1.begin(), f))
        std::cout << "Test succeeded! getCol(ci, start_ri, end_ri) methode returned expected results.\n";
    else
        std::cout << "Test failed! getCol(ci, start_ri, end_ri) methode returned unexpected results.\n";

    if ( std::equal(actual_results2.begin(), actual_results2.end(), expected_results2.begin(), f))
        std::cout << "Test succeeded! getCol(ci) methode returned expected results.\n";
    else
        std::cout << "Test failed! getCol(ci) methode returned unexpected results.\n";
    


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