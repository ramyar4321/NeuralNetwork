#include "Dataset.hpp"
#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <math.h>
#include <algorithm>


/**
 * Constructor for the Dataset class when using Haberman dataset.
 */
cpu::Dataset::Dataset(int num_rows, int num_cols, float train_test_ratio):
                      m_num_rows(num_rows),
                      m_num_cols(num_cols),
                      m_dataset(num_rows, std::vector<float>(num_cols, 0.0f)),
                      m_train_test_ratio(train_test_ratio)
{}

/**
 * Constructor for the Dataset class for train-test split.
 */
cpu::Dataset::Dataset(int num_rows, int num_cols):
                      m_num_rows(num_rows),
                      m_num_cols(num_cols),
                      m_dataset(num_rows, std::vector<float>(num_cols, 0.0f))
{}

/**
 * Constructor for Dataset object using initializer list.
 * Mainly used for testing.
 */
cpu::Dataset::Dataset(std::initializer_list< std::initializer_list<float> > ilist):
    m_dataset(ilist.begin(), ilist.end()),
    m_num_rows(ilist.size()),
    m_num_cols(ilist.begin()->size())
{}

/**
 * The function stores data from a CSV file
 * into 2d vector member variable.
 * 
 * @param filename The name of the CSV file
 */
void cpu::Dataset::import_dataset(std::string &filename){

    std::ifstream file;
    std::string line, val;   // string for line and value
    int dataset_row = 0;     // dataset row index
    int dataset_col = 0;     // dataset column index

    // Open the file and store the data into 2d vector

    file.open(filename, std::ios::in);

    if(file.is_open()){                   // Check if file was opened successfully
        while(std::getline(file, line)) { //read each line of the file
            std::stringstream s (line);   // store the line in string buffer
            while(getline(s, val, ',')){  // store each value seperated by comma delimiter into member variable 2d vector
                this->m_dataset[dataset_row][dataset_col] = std::stod(val);
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
 * Overload equality operator.
 * 
 * Two matrices are equal if and only if
 * they have the same dimensions and their
 * corresonding elements are equal.
 * 
 * return true if two matrices are equal,
 *        false otherwise
 */
bool cpu::Dataset::operator==(const cpu::Dataset& rhs) const{

    bool areEqual = true;

    // Variables to store the element of matrices to be compared
    float this_val = 0.0;
    float rhs_val = 0.0;

    // Fixed error for comparison between two given values
    constexpr double epsilon = 0.01; 

    //Check if the dimensions of the two matrices are equal
    if( this->m_num_rows != rhs.get_num_rows() ||
        this->m_num_cols != rhs.get_num_cols()){
            areEqual = false;
    }else{
        // Check if corresponding elements of the two matracies are equal
        for (int j = 0; j < this->m_num_rows; j++){
            for(int i = 0; i < this->m_num_cols; i++){
                this_val = this->m_dataset[j][i];
                rhs_val = rhs[j][i];
                if(!(std::abs(this_val - rhs_val) < epsilon)){
                    areEqual = false;
                }
            }
        }
    }

    return areEqual;

}

/**
 * Overload operator[] for read-only operation on elements of this Dataset.
 */
const std::vector<float>& cpu::Dataset::operator[](const int &input) const{
    return m_dataset[input];
}

/**
 * Overload operator[] for write operation on elements of this Dataset.
 */
std::vector<float>& cpu::Dataset::operator[](const int &input) {
    return m_dataset[input];
}

/**
 * This methode will produce the X training set
 * from the dataset. The X training set comprises 
 * all of the rows of the dataset starting from the first fow up until the train_size -1
 * row and all columns of the dataset except the outcome column.
 * 
 * @return X training set
 */
cpu::Dataset cpu::Dataset::X_train_split(){

    // Determine indices
    int train_size = static_cast<int>( this->m_num_rows * m_train_test_ratio );

    // Train data consists of all rows from
    // 0th row to (train-1)th row
    int start_ri = 0;
    int end_ri = train_size - 1;
    // train data consists of all columns
    // excluding the outcome column.
    int start_ci = 0;
    int end_ci = this->m_num_cols - 2; // Subtract 2 since there is zero indexing of vectors

    cpu::Dataset x_train = this->getSubDataset(start_ri, end_ri, start_ci, end_ci);


    return x_train; 

}

/**
 * This methode will produce the X test set from the dataset.
 * The test set comprises of all the rows starting from the
 * training size row up until the last row of the dataset,
 * and all columns except the outcome caolumn.
 * 
 */
cpu::Dataset cpu::Dataset::X_test_split(){
    int train_size = static_cast<int>( this->m_num_rows * m_train_test_ratio );

    // Train data consists of all rows from
    // train_sizeth row up until the last row.
    int start_ri = train_size;
    int end_ri = this->m_num_rows - 1; // Subtract 1 since there is zero indexing of vectors
    // train data consists of all columns
    // excluding the outcome column.
    int start_ci = 0;
    int end_ci = this->m_num_cols - 2; // Subtract 2 since there is zero indexing of vectors

    cpu::Dataset x_test = this->getSubDataset(start_ri, end_ri, start_ci, end_ci);

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
std::vector<float> cpu::Dataset::y_train_split(){

    int train_size = static_cast<int>( this->m_num_rows * m_train_test_ratio );

    int ci = 3; // Index of the outcome y column
    int start_ri = 0;
    int end_ri = train_size-1;


    std::vector<float>  y_train = this->getCol(ci, start_ri, end_ri);

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
std::vector<float> cpu::Dataset::y_test_split(){

    int train_size = static_cast<int>( this->m_num_rows * m_train_test_ratio );

    int ci = 3; // Index of the outcome y column
    int start_ri = train_size;
    int end_ri = this->m_num_rows-1;

    std::vector<float>  y_test = this->getCol(ci, start_ri, end_ri);



    return y_test;

}

/**
 * The y vector holding the outcomes of the Haberman dataset have 
 * values of 1 and 2. Since the sigmoid activation of the output neuron
 * has values between 0 and 1, it would be best to setValues of the
 * y vector to values of 0 and 1. That is, this methode will set
 * value of 1 to 0 and value 2 to 1 for the y vector holding the outcomes.
 * This will be done inplace since there is no need, for now, to make
 * waseful copies.
 * 
 * @param y A reference to the y vector storing the outcomes.
 * 
 */
void cpu::Dataset::setValues(std::vector<float>& y){

    std::replace(y.begin(), y.end(), 1, 0);
    std::replace(y.begin(), y.end(), 2, 1);

}


/**
 * Getter methode for dataset member variable.
 * 
 */
std::vector<std::vector<float> > cpu::Dataset::get_dataset(){
    return m_dataset;
}

/**
 * This methode will produce a subset, a block of entries, from the original dataset.
 * 
 * @param start_ri The index of the first row of the sub-dataset
 * @param end_ri   The index of the last row of the sub-dataset
 * @param start_ci The index of the first column of the sub-dataset
 * @param end_ci   The index of the last column of the sub-dataset
 * 
 * @return A sub-dataset containing a block of entries of the original dataset.
 * 
 */
cpu::Dataset cpu::Dataset::getSubDataset(int& start_ri, int& end_ri, int& start_ci, int& end_ci){

    // Calculate dimensions of sub-matrix
    int submat_num_rows = end_ri - start_ri + 1;
    int submat_num_cols = end_ci - start_ci + 1;

    // Create sub-matrix object to be returned
    cpu::Dataset submat(submat_num_rows, submat_num_cols);

    for(int j=0, row = start_ri; row <= end_ri; row++, j++){
        for(int i=0, col = start_ci; col <= end_ci; col++, i++){
            submat[j][i] = this->m_dataset[row][col];
        }
    }

    return submat;


}

/**
 * This methode will return all elements from row index start_ri
 * until row index end_ri for the column at index ci. 
 * @param ci       Column index of this dataset corresponding to column of interest.
 * @param start_ri Row index of this dataset corresponding to the first element 
 *                 of the column to be returned.
 * @param end_ri   Row index of this dataset corresponding to the last element 
 *                 of the column to be returned. 
 * @return If start_ri is zero and end_ri is equal to the number of rows in this matrix, 
 *         then this methode will return the column of the matrix at index ci.
 *         Otherwise, it will return a continous segment of the column at index ci. 
 */ 
 std::vector<float> cpu::Dataset::getCol(int& ci, int& start_ri, int& end_ri){


    int col_size = end_ri -start_ri +1;


    std::vector<float> col(col_size);

    for(int j= 0, row_i = start_ri; row_i <= end_ri; j++, row_i++){
        col[j] = this->m_dataset[row_i][ci];
    }

    return col;
 }

 /**
  * Return column of matrix
  * 
  * @param ci Index of the column of this dataset to be returned
  * 
  * @return Column of this dataset
  * 
  */
std::vector<float> cpu::Dataset::getCol(int& ci){
    std::vector<float> col(m_num_rows);

    for(int j = 0 ; j < m_num_rows; j++){
        col[j] = this->m_dataset[j][ci];
    }

    return col;
}

/**
 * Return row of dataset.
 * 
 * @param ri Index of the row of this dataset to be returned
 * 
 * @return Row of dataset.
 */
cpu::Vector cpu::Dataset::getRow(int& ri){
    cpu::Vector row(m_num_cols, 0.0f);


    for(int i = 0; i < m_num_cols; i++){
        row[i] = m_dataset[ri][i];
    }

    return row;
}

/**
 * Get the number of rows of this dataset.
 */
float cpu::Dataset::get_num_rows() const{
    return this->m_num_rows;
}

/**
 * Get the number of columns of this datset.
 */ 
float cpu::Dataset::get_num_cols() const{
    return this->m_num_cols;
}

/**
 * Compute the mean of values from a given column.
 * 
 * @param ci Column index for the column of interest from this dataset
 * 
 * @return mean computed for the values from the given column
 */
float cpu::Dataset::computeMean(int& ci){
    std::vector<float> col = getCol(ci);

    float sum = std::accumulate(col.begin(), col.end(), 0.0);
    float mean = sum/col.size();

    return mean;
}

/**
 * Compute the sample standard deviation for the values
 * in a given column. Standard deviation will be computed as such
 * @f$std = \sqrt{\frac{\sum_{j=0}^{n_J} (x_j - \overline{x})}{{n_J}-1}}$
 * where @f$n_J$ is the size of the given column, @f$x_j$ is an element in the 
 * given column, and @f$\overline{x}$ is the mean of for the given column.
 * 
 * Note, that computing the Standard deviation using the following formula
 * @f$std = \sqrt{\frac{\sum_{j=0}^{n_J} (x_j)^2}{{n_J}-1}} -\overline{x}^2$
 * is more prone to overflow or underflow, thus it will not be used here.
 * 
 * @param col A given column from a matrix.
 * 
 * @return Standard deviation for the given column
 */
float cpu::Dataset::computeStd(int& ci){
    std::vector<float> col = getCol(ci);

    float mean  = computeMean(ci);

    float accum = 0.0;
    std::for_each(col.begin(), col.end(), [&](const float x) {
    accum += (x - mean) * (x - mean);
    });

    float std = static_cast<float>( sqrt(accum/(col.size() -1)));

    return std;
}

/**
 * Compute the sample standard deviation for the values
 * in a given column. Standard deviation will be computed as such
 * @f$std = \sqrt{\frac{\sum_{j=0}^{n_J} (x_j - \overline{x})}{{n_J}-1}}$
 * where @f$n_J$ is the size of the given column, @f$x_j$ is an element in the 
 * given column, and @f$\overline{x}$ is the mean of for the given column.
 * 
 * Note, that computing the Standard deviation using the following formula
 * @f$std = \sqrt{\frac{\sum_{j=0}^{n_J} (x_j)^2}{{n_J}-1}} -\overline{x}^2$
 * is more prone to overflow or underflow, thus it will not be used here.
 * 
 * @param col AThe column for which we want the standard deviation
 * @param mean The column for which we want the mean
 * 
 * @return Standard deviation for the given column
 */
float cpu::Dataset::computeStd(int& ci, float& mean){
    std::vector<float> col = getCol(ci);

    float accum = 0.0;
    std::for_each(col.begin(), col.end(), [&](const float x) {
    accum += (x - mean) * (x - mean);
    });

    float std = static_cast<float>( sqrt(accum/(col.size() -1)));

    return std;
}

/**
 * Rescale the data to have a mean of 0 and standard deviation of 1.
 * More percisely, compute the z-score for each element of the matrix.
 * 
 * @return A matrix containing the z-score for each element of this matrix
 * 
 */
cpu::Dataset cpu::Dataset::standardizeDataset(){
    float col_mean = 0.0;
    float col_std= 0.0;

    cpu::Dataset stand_mat(this->m_num_rows, this->m_num_cols);

    for(int i=0; i < this->m_num_cols; i++){
        col_mean = this->computeMean(i);
        col_std = this->computeStd(i);
        for(int j=0; j < this->m_num_rows; j++){
            stand_mat[j][i] = (static_cast<float>(this->m_dataset[j][i]) - col_mean)/col_std;
        }
    }

    return stand_mat;

}