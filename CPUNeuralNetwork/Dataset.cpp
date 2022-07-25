#include "Dataset.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>


/**
 * Constructor for the Dataset class.
 */
cpu::Dataset::Dataset(int dataset_col_size, 
                      int dataset_row_size,
                      float train_test_ratio):
                      m_dataset(dataset_row_size, dataset_col_size),
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
 * all of the rows of the dataset starting from the first fow up until the train_size -1
 * row and all columns of the dataset except the outcome column.
 * 
 * @return X training set
 */
cpu::Matrix cpu::Dataset::X_train_split(){

    // Determine indices
    int train_size = static_cast<int>( m_dataset.get_num_rows() * m_train_test_ratio );

    // Train data consists of all rows from
    // 0th row to (train-1)th row
    int start_ri = 0;
    int end_ri = train_size - 1;
    // train data consists of all columns
    // excluding the outcome column.
    int start_ci = 0;
    int end_ci = m_dataset.get_num_cols() - 2; // Subtract 2 since there is zero indexing of vectors

    Matrix x_train = m_dataset.getSubMatrix(start_ri, end_ri, start_ci, end_ci);


    return x_train; 

}

/**
 * This methode will produce the X test set from the dataset.
 * The test set comprises of all the rows starting from the
 * training size row up until the last row of the dataset,
 * and all columns except the outcome caolumn.
 * 
 */
cpu::Matrix cpu::Dataset::X_test_split(){
    int train_size = static_cast<int>( m_dataset.get_num_rows() * m_train_test_ratio );

    // Train data consists of all rows from
    // train_sizeth row up until the last row.
    int start_ri = train_size;
    int end_ri = m_dataset.get_num_rows() - 1; // Subtract 1 since there is zero indexing of vectors
    // train data consists of all columns
    // excluding the outcome column.
    int start_ci = 0;
    int end_ci = m_dataset.get_num_cols() - 2; // Subtract 2 since there is zero indexing of vectors

    Matrix x_test = m_dataset.getSubMatrix(start_ri, end_ri, start_ci, end_ci);

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

    int train_size = static_cast<int>( m_dataset.get_num_rows() * m_train_test_ratio );

    int ci = 3; // Index of the outcome y column
    int start_ri = 0;
    int end_ri = train_size-1;


    std::vector<double>  y_train = m_dataset.getCol(ci, start_ri, end_ri);

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

    int train_size = static_cast<int>( m_dataset.get_num_rows() * m_train_test_ratio );

    int ci = 3; // Index of the outcome y column
    int start_ri = train_size;
    int end_ri = m_dataset.get_num_rows()-1;

    std::vector<double>  y_test = m_dataset.getCol(ci, start_ri, end_ri);



    return y_test;

}




/**
 * Getter methode for dataset member variable.
 * 
 */
cpu::Matrix cpu::Dataset::get_dataet(){
    return m_dataset;
}

