#include "DatasetTesting.hpp"
#include "../Dataset.hpp"
#include <iostream>
#include <functional>

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
void cpu::DatasetTesting::test_import_dataset(){

    // Instantiate Dataset object and call methode to be tested.
    cpu::Dataset dat(306, 4, 0.75);
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    std::vector<std::vector<float> > dataset = dat.get_dataset();

    // The actual values
    float actual_value1 = dataset[0][0];
    float actual_value2 = dataset[0][1];
    float actual_value3 = dataset[0][2];
    float actual_value4 = dataset[0][3];

    float actual_value5 = dataset[305][0];
    float actual_value6 = dataset[305][1];
    float actual_value7 = dataset[305][2];
    float actual_value8 = dataset[305][3];

    // The expected values
    float expected_value1 = 30;
    float expected_value2 = 64;
    float expected_value3 = 1;
    float expected_value4 = 1;

    float expected_value5 = 83;
    float expected_value6 = 58;
    float expected_value7 = 2;
    float expected_value8 = 2;

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
void cpu::DatasetTesting::test_X_train_split(){

    // Instantiate objects and initialize variables

    // Instantiate Dataset object
    // Train test ratio will be set to 0.99.
    cpu::Dataset dat(306, 4, 0.99);
    //Import the dataset
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    std::vector<std::vector<float> > dataset = dat.get_dataset();

    // Call methode to be tested.
    cpu::Dataset train_dataset = dat.X_train_split();

    // Actual size of the X train set data
    int actual_col_size = train_dataset.get_num_cols();
    int actual_row_size = train_dataset.get_num_rows();
 
    // Expected size of the X train set data.
    // The expected number of columns is 3 since
    // the training set includes all the columns of the dataset except 
    // the outcome column.
    int expected_col_size = 3;
    // The expected number of rows is 
    // floor((train test ratio)*(number of rows of the dataset)) = floor(0.99*306) = 302 
    int expected_row_size = 302;

    // The actual values of the first row of the X train dataset
    float actual_value1 = train_dataset[0][0];
    float actual_value2 = train_dataset[0][1];
    float actual_value3 = train_dataset[0][2];

    // The actual values of the last row of the X train dataset
    float actual_value4 = train_dataset[301][0];
    float actual_value5 = train_dataset[301][1];
    float actual_value6 = train_dataset[301][2];

    // The expected values for the first row of the X train dataset
    // are the values for the first row of the dataset.
    float expected_value1 = 30;
    float expected_value2 = 64;
    float expected_value3 = 1;

    // The expected values for the last row of the X train dataset
    // are the values for the train_sizeth row of the dataset.
    float expected_value4 = 75;
    float expected_value5 = 62;
    float expected_value6 = 1;

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
void cpu::DatasetTesting::test_X_test_split(){

    // Instantiate objects and initialize variables

    // Instantiate Dataset object
    // Train test ratio will be set to 0.99.
    cpu::Dataset dat(306, 4, 0.99);
    //Import the dataset
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    std::vector<std::vector<float> > dataset = dat.get_dataset();

    // Call methode to be tested.
    cpu::Dataset test_dataset = dat.X_test_split();

    // Actual size of the X test set data
    int actual_col_size = test_dataset.get_num_cols();
    int actual_row_size = test_dataset.get_num_rows();
 
    // Expected size of the X test set data.
    // The expected number of columns is 3 since
    // the test set includes all the columns of the dataset except 
    // the outcome column.
    int expected_col_size = 3;
    // The expected number of rows is 
    // ceiling((1- train test ratio)*(number of rows of the dataset)) = ceiling(0.01*306) = 4
    int expected_row_size = 4;

    // The actual values of the first row of the X test dataset
    float actual_value1 = test_dataset[0][0];
    float actual_value2 = test_dataset[0][1];
    float actual_value3 = test_dataset[0][2];

    // The actual values of the last row of the X test dataset
    float actual_value4 = test_dataset[3][0];
    float actual_value5 = test_dataset[3][1];
    float actual_value6 = test_dataset[3][2];

    // The expected values for the first row of the X test dataset
    // are the values for the train_sizeth + 1 row of the dataset.
    float expected_value1 = 76;
    float expected_value2 = 67;
    float expected_value3 = 0;

    // The expected values for the last row of the X test dataset
    // are the values for the last row of the dataset.
    float expected_value4 = 83;
    float expected_value5 = 58;
    float expected_value6 = 2;

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
void cpu::DatasetTesting::test_y_train_split(){

    // Instantiate objects and initialize variables

    // Instantiate Dataset object
    // Train test ratio will be set to 0.99.
    cpu::Dataset dat(306, 4, 0.99);
    //Import the dataset
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    std::vector<std::vector<float> > dataset = dat.get_dataset();

    // Call methode to be tested.
    std::vector<float> ytrain_dataset = dat.y_train_split();

    // Actual size of the y train vector
    int actual_size = ytrain_dataset.size();
 
    // The expected size of y train vector.
    // floor((train test ratio)*(number of rows of the dataset)) = floor(0.99*306) = 302
    int expected_size = 302;

    // The first actual value of y train vector.
    float actual_value1 = ytrain_dataset[0];

    // The last actual value of the y train vector.
    float actual_value2 = ytrain_dataset[301];

    // The expected value for the first element of y train vector
    // is the value of the first element of the outcome from the dataset.
    float expected_value1 = 1;
    
    // The expected value for the last element of y train vector
    // is the value of the train_sizeth element of the outcome from the dataset.
    float expected_value2 = 1;


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
void cpu::DatasetTesting::test_y_test_split(){

    // Instantiate objects and initialize variables

    // Instantiate Dataset object
    // Train test ratio will be set to 0.99.
    cpu::Dataset dat(306, 4, 0.99);
    //Import the dataset
    std::string filename = "../Data/haberman.data";
    dat.import_dataset(filename);
    std::vector<std::vector<float> > dataset = dat.get_dataset();

    // Call methode to be tested.
    std::vector<float> ytest_dataset = dat.y_test_split();

    // Actual size of the y test vector
    int actual_size = ytest_dataset.size();
 
    // The expected size of y test vector.
    // ceiling((1 - train test ratio)*(number of rows of the dataset)) = floor(0.01*306) = 4
    int expected_size = 4;

    // The first actual value of y test vector.
    float actual_value1 = ytest_dataset[0];

    // The last actual value of the y test vector.
    float actual_value2 = ytest_dataset[3];

    // The expected value for the first element of y test vector
    // is the value of the train_size+1 element of the outcome from the dataset.
    float expected_value1 = 1;
    
    // The expected value for the last element of y test vector
    // is the value of the last element of the outcome from the dataset.
    float expected_value2 = 2;


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
void cpu::DatasetTesting::test_getColumn(){
    
    // Matrix used for testing
    cpu::Dataset dat = {{1,2,3,4},
                  {5,6,7,8},
                  {9,10,11,12},
                  {13,14,15,16}};

    // Test if the first column, considering zero indexing, was retreived
    int ci = 1;

    int start_ri = 0;
    int end_ri = 2;
    std::vector<float> actual_results1 = dat.getCol(ci, start_ri, end_ri);
    std::vector<float> expected_results1 = {2,6,10};

    
    std::vector<float> actual_results2 = dat.getCol(ci);
    std::vector<float> expected_results2 = {2,6,10,14};

    // Function pointer to helper function to be used as callback function
    // when comparing actual and expected values.
    std::function<bool(float,float)> f = &cpu::DatasetTesting::areFloatEqual;

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
 * Test the getRow methode of the Dataset class
 */
void cpu::DatasetTesting::test_getRow(){
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
    std::function<bool(float,float)> f = &cpu::DatasetTesting::areFloatEqual;

    if ( actual_results == expected_results)
        std::cout << "Test succeeded! getRow methode returned expected results.\n";
    else
        std::cout << "Test failed! getROw methode returned unexpected results.\n";

}

/**
 * This methode tests the setValue methode of the Dataset class.
 */
void cpu::DatasetTesting::test_setValue(){

    // Instantiated Dataset object.
    // The parameters are not important. We smiply
    // need an object to access the setValue methode.
    cpu::Dataset dat(306, 4, 0.99);

    std::vector<float> y_actual = {2,1,2,1};

    std::vector<float> y_expect = {1,0,1,0};

    dat.setValues(y_actual);

    // Function pointer to helper function to be used as callback function
    // when comparing actual and expected values.
    std::function<bool(float,float)> f = &cpu::DatasetTesting::areFloatEqual;

    // Test of setValue returned expected results.
    if ( std::equal(y_actual.begin(), y_actual.end(), y_expect.begin(),f))
        std::cout << "Test succeeded! setValue methode returned expected results.\n";
    else
        std::cout << "Test failed! setValue methode returned unexpected results.\n";

}
  
/**
 * Test standardizeDataset of the Dataset class.
 * computeStd and computeMean are also tested since
 * standardizeDataset relies upon them.
 */
void cpu::DatasetTesting::test_standardizeDataset(){

    
    cpu::Dataset dat = {{1,2,3},  // Matrix used for testing
                  {4,5,6},
                  {7,8,9},
                  {10,11,12}};

    cpu::Dataset actual_result = dat.standardizeDataset();

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
// Helper methodes.

/**
 * Determine if two float values are equal with a fixed error. 
 * Fixed point errors are not used for comparison between floating point values
 * but it will suffice for our usage. 
 */
bool cpu::DatasetTesting::areFloatEqual(float a, float b){
    constexpr float epsilon = 0.01; 
    return std::abs(a - b) < epsilon;
}