#include "Dataset.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>

/**
 * Constructor for the Dataset class.
 */
cpu::Dataset::Dataset(unsigned int dataset_col_size, 
                      unsigned int dataset_row_size,
                      float train_test_ratio):
                      m_dataset(dataset_row_size, std::vector<double>(dataset_col_size)),
                      m_train_test_ratio(train_test_ratio)
{}

/**
 * The function stores data from a CSV file
 * into 2d vector memeber variable.
 * 
 * @param filename The name of the CSV file
 */
void cpu::Dataset::import_dataset(std::string &filename){

    std::ifstream file;
    std::string line, val;   // string for line & value
    int dataset_row = 0;     // dataset row index
    int dataset_col = 0;     // dataset column index

    // Open the file and store the data into 2d vector

    file.open(filename, std::ios::in);

    if(file.is_open()){                   // Check if file was opened successfully
        while(std::getline(file, line)) { //read each line of the file
            std::stringstream s (line);   // store the line in string buffer
            while(getline(s, val, ',')){  // store each value seperated by comma delimiter into member variable 2d vector
                m_dataset[dataset_row][dataset_col] = std::stod(val);
                dataset_col++;
            }
            dataset_col = 0;
            dataset_row++;
        }
    } else{
        std::cerr << "Error unable to open file";
    }
}

/**
 * This methode will produce the X training set
 * from the dataset. The X training set comprises 
 * all of the rows of the dataset starting from the first fow up until the train_size
 * row and all columns of the dataset except the outcome column.
 * 
 * @return X training set
 */
std::vector<std::vector<double> > cpu::Dataset::X_train_split(){

    int train_size = static_cast<int>( m_dataset.size() * m_train_test_ratio );

    std::vector<std::vector<double> > x_train(m_dataset.begin(), m_dataset.begin() + train_size);

    // Remove the outcome column which is the last column of the dataset.
    for(auto &row: x_train){
        row.erase(std::next(row.begin(), 3));
    }


    return x_train;

}

/**
 * This methode will produce the X test set from the dataset.
 * The test set comprises of all the rows starting from the
 * training size + 1 row up until the last row of the dataset,
 * and all columns except the outcome caolumn.
 * 
 */
std::vector<std::vector<double> > cpu::Dataset::X_test_split(){
    int test_size = static_cast<int>( m_dataset.size() * m_train_test_ratio);

    std::vector<std::vector<double> > x_test(m_dataset.begin() + test_size, m_dataset.end());

    // Remove the outcome column which is the last column of the dataset
    for(auto &row: x_test){
        row.erase(std::next(row.begin(), 3));
    }

    return x_test;
}

/**
 * This methode will produce the y training set
 * from the dataset. The y training set comprises of
 * all of the outcomes of the dataset starting 
 * from the first row up until the train_sizeth row.
 * 
 * @return y training set
 */
std::vector<double> cpu::Dataset::y_train_split(){

    int train_size = static_cast<int>( m_dataset.size() * m_train_test_ratio );

    std::vector<double>  y_train;
    y_train.reserve(train_size);

    // Extract outcome column from dataset into y train vector
    std::transform(m_dataset.begin(), m_dataset.begin() + train_size, std::back_inserter(y_train),
                    [](const std::vector<double> &row) {return row[3];});

    return y_train;

}

/**
 * This methode will produce the y testing set
 * from the dataset. The y testing set comprises of
 * all of the outcomes of the dataset starting 
 * from the train_sizeth row up until the last row.
 * 
 * @return y test set
 */
std::vector<double> cpu::Dataset::y_test_split(){

    int train_size = static_cast<int>( m_dataset.size() * m_train_test_ratio );

    int test_size = static_cast<int>(m_dataset.size()) - train_size;

    // Create y test vector
    std::vector<double> y_test;
    y_test.reserve(test_size);

    // Extract outcome column from dataset into y test vector
    std::transform(m_dataset.begin() + train_size, m_dataset.end(), std::back_inserter(y_test),
                    [](const std::vector<double> &row) {return row[3];});


    return y_test;

}

/**
 * Getter methode for dataset member variable.
 * 
 */
std::vector<std::vector<double> > cpu::Dataset::get_dataet(){
    return m_dataset;
}

