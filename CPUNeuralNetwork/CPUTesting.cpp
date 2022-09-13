#include "CPUTesting.hpp"
#include "CPUNeuralNetwork.hpp"
#include "Dataset.hpp"
#include <vector>
#include <iostream>
#include <functional>


/*----------------------------------------------*/
// Testing methodes of the NeuralNetowrk class

/**
 * This methodes tests forward propegation of the Neural Network.
 */
void cpu::Testing::test_forwardPropegation(){

    cpu::Vector x = {1.0f};
    cpu::Matrix W1(2,1,{1.0f, 0.0f});
    cpu::Matrix W2(2,2,{1.0f, 0.0f, 0.0f, 0.0f});
    cpu::Matrix W3 = {1, 2, {1.0f, 0.0f}};

    cpu::NeuralNetwork net(0, 0.01);



    cpu::Layer* hiddenlayer1 = new HiddenLayer(1,2);
    cpu::Layer* hiddenlayer2 = new HiddenLayer(2,2);
    cpu::Layer* outputlayer = new OutputLayer(2,1);

    net.addLayer(hiddenlayer1);
    net.addLayer(hiddenlayer2);
    net.addLayer(outputlayer);

    net.x(x);
    net.W(W1, 0);
    net.W(W2, 1);
    net.W(W3, 2);
    net.x(x);

    cpu::Vector actual_a3 = net.forwardPropegation();

    cpu::Vector expected_a3 = {0.731f};



    if( expected_a3 == actual_a3){
        std::cout << "Test passed! Forward propegation produced expected results." << std::endl;
    }else{
        std::cout << "Test failed! Forward propegation produced unexpected results." << std::endl;
    }
}

/**
 * 
 * This methode tests if the correct gradient is computed.
 * The finite difference will be used to approximate the expected gradient. 
 * 
 * The algorithm is as follows:
 * 1. Radomly initialize weight of neural network from a Guassian distribution. 
 * 2. Perform forward and backpropegation to determine the gradients computed
 *    for the last layer of the neural network and store the result. 
 * 3. for each layer:
 *          For each gradient w of the last weight:
 *              - compute negative perturbation: w_minus = w- perturb
 *              - perform forward propegation of neural network 
 *                with w_minus instead of w
 *              - compute loss_minus which is the loss of the neural network
 *                 by replacing w with w_minus.
 *              - compute positve pertubation: w_positive = w + perturb
 *              - perform forward propegation of neural network with 
 *                w_positive instead of w
 *              - compute loss_positive which is the loss of the neural network
 *                by replacing w with w_positive.
 *              - estimate numerical gradient for w by using the finite difference methode
 *                numericGradient = (loss_positive - loss_negative)/2*perturb
 *              - store numericGradient.
 * 4. Compare the gradients for the last layer produced by backpropegation
 *    with the numerical estimated gradients.
 */
void cpu::Testing::test_backPropegation(){

    /*cpu::NeuralNetwork net(10,10, 0, 0.01);

    cpu::Vector x = {-2.11764, 0.3571 , -0.423171};
    double y = 0;

    cpu::Matrix W1(10,3);
    cpu::Matrix W2(10,10);
    cpu::Vector W3(10,0.0f);


    cpu::Matrix W1_minus(10,3);
    cpu::Matrix W1_plus(10,3);

    cpu::Matrix W2_minus(10,10);
    cpu::Matrix W2_plus(10,10);

    cpu::Vector W3_minus(10, 0.0f);
    cpu::Vector W3_plus(10, 0.0f);

    cpu::Matrix numericdLdW1(10,3);
    cpu::Matrix numericdLdW2(10,10);
    cpu::Vector numericdLdW3(10, 0.0f);

    double perturb = 0.00000001;

    double loss_minus;
    double loss_plus;

    W1.matrixInitialization();
    W2.matrixInitialization();
    W3.vectorInitialization();

    net.m_x = x;
    net.m_hidden_layer1.m_W = W1;
    net.m_hidden_layer2.m_W = W2;
    net.m_output_layer.m_W = W3;
    net.m_y = y;

    net.forwardPropegation();
    net.backPropegation();

    cpu::Vector actual_dLdW3 = net.m_output_layer.m_dLdW;
    cpu::Matrix actual_dLdW2 = net.m_hidden_layer2.m_dLdW;
    cpu::Matrix actual_dLdW1 = net.m_hidden_layer1.m_dLdW;

    for(int i=0; i < W3.getSize(); i++){
        W3_minus = W3;
        W3_plus = W3;
        W3_minus[i] -= perturb;
        W3_plus[i] += perturb;

        net.m_output_layer.m_W = W3_minus;
        net.forwardPropegation();
        loss_minus = net.m_output_layer.computeLoss(y);

        net.m_output_layer.m_W = W3_plus;
        net.forwardPropegation();
        loss_plus =net.m_output_layer.computeLoss(y);

        numericdLdW3[i] = (loss_plus-loss_minus)/(2*perturb);      
    }
    net.m_output_layer.m_W = W3;

    for (int j = 0; j < W2.get_num_rows(); j++){
        for(int i=0; i < W2.get_num_cols(); i++){
            W2_minus = W2;
            W2_plus = W2;
            W2_minus(j,i) -= perturb;
            W2_plus(j,i) += perturb;

            net.m_hidden_layer2.m_W = W2_minus;
            net.forwardPropegation();
            loss_minus = net.m_output_layer.computeLoss(y);

            net.m_hidden_layer2.m_W = W2_plus;
            net.forwardPropegation();
            loss_plus = net.m_output_layer.computeLoss(y);

            numericdLdW2(j,i) = (loss_plus-loss_minus)/(2*perturb);
        }
    }
    net.m_hidden_layer2.m_W = W2;

    for (int j = 0; j < W1.get_num_rows(); j++){
        for(int i=0; i < W1.get_num_cols(); i++){
            W1_minus = W1;
            W1_plus = W1;
            W1_minus(j,i) -= perturb;
            W1_plus(j,i) += perturb;

            net.m_hidden_layer1.m_W = W1_minus;
            net.forwardPropegation();
            loss_minus = net.m_output_layer.computeLoss(y);

            net.m_hidden_layer1.m_W = W1_plus;
            net.forwardPropegation();
            loss_plus =net.m_output_layer.computeLoss(y);

            numericdLdW1(j,i) = (loss_plus-loss_minus)/(2*perturb);
        }
    }

    if ( actual_dLdW3 == numericdLdW3)
        std::cout << "Test succeeded! Backpropegation gradient matches numeric gradient for last layer.\n";
    else
        std::cout << "Test failed! Backpropegation gradient does not match numeric gradient for last layer.\n";
    
    if(actual_dLdW2 == numericdLdW2){
        std::cout << "Test succeeded! Backpropegation gradient matches numeric gradient for second layer.\n";
    } else{
        std::cout << "Test failed! Backpropegation gradient does not match numeric gradient for second layer.\n";
    }

    if(actual_dLdW1 == numericdLdW1){
        std::cout << "Test succeeded! Backpropegation gradient matches numeric gradient for first layer.\n";
    } else{
        std::cout << "Test failed! Backpropegation gradient does not match numeric gradient for first layer.\n";
    }*/
}

/**
 * This methode tests the gradient descent algorithm the derived layers classes.
 * Since both the hidden and output layers shared the same algorithm, only the
 * output layer gradientDescent methode will be test.
 * 
 * The gradient decent methode must produce a series of 
 * non-decreasing objectives in order for this test to pass.
 * The details of this test is as follows.
 * Let @f$L = w^2$ and then @f$dLdw = 2w$ with step size of @f$\alpha = 0.01$.
 * The initial starting position will be @f$w=100$ and the number of iterations
 * will be 5. The choices of these numbers are random. If after each iteration,
 * the loss for the new position is smaller than the loss for the old position,
 * then this test will pass.
 * 
 */
void cpu::Testing::test_gradientDescent(){

    /*cpu::OutputLayer outputlayer(1);
    double alpha = 0.01;

    bool testPass = true;

    int numIter = 5;

    cpu::Vector w(1, 100);
    outputlayer.m_W = w;
    double loss = computeQuadraticLoss(w);
    double prev_loss;
    cpu::Matrix dLdw = computeGradientQuadraticLoss(w);

    for(int i = 0; i < numIter; i++){
        prev_loss = loss;
        outputlayer.gradientDecent(alpha);
        loss = computeQuadraticLoss(w);
        if(loss > prev_loss){
            testPass = false;
        }
    }

    if(testPass){
        std::cout << "Test succeeded! Gradient descent produces expected results." << std::endl;
    }else{
        std::cout << "Test failed! Gradient descent produces unexpected results." << std::endl;
    }*/
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
    cpu::Dataset dat(306, 4, 0.75);
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    std::vector<std::vector<double> > dataset = dat.get_dataset();

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
    cpu::Dataset dat(306, 4, 0.99);
    //Import the dataset
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    std::vector<std::vector<double> > dataset = dat.get_dataset();

    // Call methode to be tested.
    cpu::Dataset train_dataset = dat.X_train_split();

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
    cpu::Dataset dat(306, 4, 0.99);
    //Import the dataset
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    std::vector<std::vector<double> > dataset = dat.get_dataset();

    // Call methode to be tested.
    cpu::Dataset test_dataset = dat.X_test_split();

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
    cpu::Dataset dat(306, 4, 0.99);
    //Import the dataset
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    std::vector<std::vector<double> > dataset = dat.get_dataset();

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
    cpu::Dataset dat(306, 4, 0.99);
    //Import the dataset
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    std::vector<std::vector<double> > dataset = dat.get_dataset();

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
 * This methode will test both overload getColumn methodes.
 * There will be two tests, one for each overload methode.
 */
void cpu::Testing::test_getColumn(){
    
    // Matrix used for testing
    cpu::Dataset dat = {{1,2,3,4},
                  {5,6,7,8},
                  {9,10,11,12},
                  {13,14,15,16}};

    // Test if the first column, considering zero indexing, was retreived
    int ci = 1;

    int start_ri = 0;
    int end_ri = 2;
    std::vector<double> actual_results1 = dat.getCol(ci, start_ri, end_ri);
    std::vector<double> expected_results1 = {2,6,10};

    
    std::vector<double> actual_results2 = dat.getCol(ci);
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

/**
 * This methode tests the setValue methode of the Dataset class.
 */
void cpu::Testing::test_setValue(){

    // Instantiated Dataset object.
    // The parameters are not important. We smiply
    // need an object to access the setValue methode.
    cpu::Dataset dat(306, 4, 0.99);

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
    cpu::Dataset dat = {{1,2,3},
                  {4,5,6},
                  {7,8,9}};

    int ci = 0;

    // Test to see if zeroth column, considering zero indexing, mean was correctly computed.
    double actual_result= dat.computeMean(ci);
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
    cpu::Dataset dat = {{1,2,3},
                  {4,5,6},
                  {7,8,9}};

    int ci = 0;

    // Test to determine if both computeStd methodes
    // compute the correct Standard deviation for the zeroth column.
    double actual_result1= dat.computeStd(ci);

    double mean  = dat.computeMean(ci);
    double actual_result2 = dat.computeStd(ci, mean);
    
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

    
    cpu::Dataset dat = {{1,2,3},  // Matrix used for testing
                  {4,5,6},
                  {7,8,9},
                  {10,11,12}};

    cpu::Dataset actual_result = dat.standardizeMatrix();

    cpu::Dataset expected_result = {{-1.1619,-1.1619,-1.1619},
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
    cpu::Dataset dat = {{1,2,3},
                  {4,5,6},
                  {7,8,9}};

    // Test to see if first row, considering zero indexing, was retreived.
    int ri = 1;
    cpu::Vector actual_results = dat.getRow(ri);
    cpu::Vector expected_results = {4,5,6};

    // Function pointer to helper function to be used as callback function
    // when comparing actual and expected values.
    std::function<bool(double,double)> f = &cpu::Testing::areFloatEqual;

    if ( actual_results == expected_results)
        std::cout << "Test succeeded! getRow methode returned expected results.\n";
    else
        std::cout << "Test failed! getROw methode returned unexpected results.\n";

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

/**
 * Compute and return the quadratic loss as follows.
 * @f$L = w^2$
 * 
 */
double cpu::Testing::computeQuadraticLoss(cpu::Vector& w){
    double quadraticLoss = w[0]*w[0];

    return quadraticLoss;
}

/**
 * 
 * Compute and return the derivative of the
 * quadratic loss function.
 * 
 */
cpu::Matrix cpu::Testing::computeGradientQuadraticLoss(cpu::Vector& w){
    double gradientQuadracticLoss = 2*w[0];

    cpu::Matrix gradientQuadracticLoss_(1,1,{gradientQuadracticLoss});

    return gradientQuadracticLoss_;
}