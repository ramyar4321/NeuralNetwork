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
                      m_dataset(num_rows, std::vector<double>(num_cols, 0.0f)),
                      m_train_test_ratio(train_test_ratio)
{}

/**
 * Constructor for the Dataset class for train-test split.
 */
cpu::Dataset::Dataset(int num_rows, int num_cols):
                      m_num_rows(num_rows),
                      m_num_cols(num_cols),
                      m_dataset(num_rows, std::vector<double>(num_cols, 0.0f))
{}

/**
 * Constructor for Dataset object using initializer list.
 * Mainly used for testing.
 */
cpu::Dataset::Dataset(std::initializer_list< std::initializer_list<double> > ilist):
    m_dataset(ilist.begin(), ilist.end()),
    m_num_rows(ilist.size()),
    m_num_cols(ilist.begin()->size())
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
 * This methode will produce the X training set
 * from the dataset. The X training set comprises 
 * all of the rows of the dataset starting from the first fow up until the train_size -1
 * row and all columns of the dataset except the outcome column.
 * 
 * @return X training set
 */
cpu::Matrix cpu::Dataset::X_train_split(){

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

    Matrix x_train = this->getSubDataset(start_ri, end_ri, start_ci, end_ci);


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
    int train_size = static_cast<int>( this->m_num_rows * m_train_test_ratio );

    // Train data consists of all rows from
    // train_sizeth row up until the last row.
    int start_ri = train_size;
    int end_ri = this->m_num_rows - 1; // Subtract 1 since there is zero indexing of vectors
    // train data consists of all columns
    // excluding the outcome column.
    int start_ci = 0;
    int end_ci = this->m_num_cols - 2; // Subtract 2 since there is zero indexing of vectors

    Matrix x_test = this->getSubDataset(start_ri, end_ri, start_ci, end_ci);

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

    int train_size = static_cast<int>( this->m_num_rows * m_train_test_ratio );

    int ci = 3; // Index of the outcome y column
    int start_ri = 0;
    int end_ri = train_size-1;


    std::vector<double>  y_train = this->getCol(ci, start_ri, end_ri);

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

    int train_size = static_cast<int>( this->m_num_rows * m_train_test_ratio );

    int ci = 3; // Index of the outcome y column
    int start_ri = train_size;
    int end_ri = this->m_num_rows-1;

    std::vector<double>  y_test = this->getCol(ci, start_ri, end_ri);



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
void cpu::Dataset::setValues(std::vector<double>& y){

    std::replace(y.begin(), y.end(), 1, 0);
    std::replace(y.begin(), y.end(), 2, 1);

}


/**
 * Getter methode for dataset member variable.
 * 
 */
std::vector<std::vector<double> > cpu::Dataset::get_dataset(){
    return m_dataset;
}

/**
 * This methode will produce a submatrix, a block of entries from the original dataset.
 * 
 * @param start_ri The index of the first row of the sub-matrix
 *                 0 <= start_ri < (number of rows in orginal dataset)
 * @param end_ri   The index of the last row of the sub-matrix
 *                 0 <= end_ri < (number of rows in orginal dataset)
 * @param start_ci The index of the first column of the sub-matrix
 *                 0 <= start_ci  < (number of columns in the original dataset)
 * @param end_ci   The index of the last column of the sub-matrix
 *                 0 <= end_ci  < (number of columns in the original dataset)
 * 
 * @return A sub-matrix containing a block of entries of the original dataset.
 * 
 */
cpu::Matrix cpu::Dataset::getSubDataset(int& start_ri, int& end_ri, int& start_ci, int& end_ci){

    // Assert that Matrix indices are withing the dimensions of this Matrix
    assert(start_ri >= 0 && start_ri < m_num_rows);
    assert(end_ri >= 0 && end_ri < m_num_rows);
    assert(start_ci >= 0 && start_ci < m_num_cols);
    assert(start_ci >= 0 && start_ci < m_num_cols);

    // Calculate dimensions of sub-matrix
    int submat_num_rows = end_ri - start_ri + 1;
    int submat_num_cols = end_ci - start_ci + 1;

    // Create sub-matrix object to be returned
    cpu::Matrix submat(submat_num_rows, submat_num_cols);

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
 * @param ci       Column index of this matrix corresponding
 * @param start_ri Row index of this matrix corresponding to the first element 
 *                 of the column to be returned.
 * @param end_ri   Row index of this matrix corresponding to the last element 
 *                 of the column to be returned. 
 * @return If start_ri is zero and end_ri is equal to the number of rows in this matrix, 
 *         then this methode will return the column of the matrix at index ci.
 *         Otherwise, it will return a continous segment of the column at index ci. 
 */ 
 std::vector<double> cpu::Dataset::getCol(int& ci, int& start_ri, int& end_ri){
    assert(start_ri >= 0 && start_ri < m_num_rows);
    assert(end_ri >= 0 && end_ri < m_num_rows);
    assert(ci >= 0 && ci < m_num_cols);

    int col_size = end_ri -start_ri +1;


    std::vector<double> col(col_size);

    for(int j= 0, row_i = start_ri; row_i <= end_ri; j++, row_i++){
        col[j] = this->m_dataset[row_i][ci];
    }

    return col;
 }

 /**
  * Return column of matrix
  * 
  * @param ci Index of the column of this matrix to be returned
  * 
  * @return Column of matrix
  * 
  */
std::vector<double> cpu::Dataset::getCol(int& ci){
    std::vector<double> col(m_num_rows);

    for(int j = 0 ; j < m_num_rows; j++){
        col[j] = this->m_dataset[j][ci];
    }

    return col;
}


/**
 * Compute the mean of values from a given column.
 * 
 * @param ci Column index for the column of interest from this matrix
 * 
 * @return mean computed for the values from the given column
 */
double cpu::Dataset::computeMean(int& ci){
    std::vector<double> col = getCol(ci);

    double sum = std::accumulate(col.begin(), col.end(), 0.0);
    double mean = sum/col.size();

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
double cpu::Dataset::computeStd(int& ci){
    std::vector<double> col = getCol(ci);

    double mean  = computeMean(ci);

    double accum = 0.0;
    std::for_each(col.begin(), col.end(), [&](const double x) {
    accum += (x - mean) * (x - mean);
    });

    double std = sqrt(accum/(col.size() -1));

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
double cpu::Dataset::computeStd(int& ci, double& mean){
    std::vector<double> col = getCol(ci);

    double accum = 0.0;
    std::for_each(col.begin(), col.end(), [&](const double x) {
    accum += (x - mean) * (x - mean);
    });

    double std = sqrt(accum/(col.size() -1));

    return std;
}

/**
 * Rescale the data to have a mean of 0 and standard deviation of 1.
 * More percisely, compute the z-score for each element of the matrix.
 * 
 * @return A matrix containing the z-score for each element of this matrix
 * 
 */
cpu::Matrix cpu::Dataset::standardizeMatrix(){
    double col_mean = 0.0;
    double col_std= 0.0;

    Matrix stand_mat(this->m_num_rows, this->m_num_cols);

    for(int i=0; i < this->m_num_cols; i++){
        col_mean = this->computeMean(i);
        col_std = this->computeStd(i);
        for(int j=0; j < this->m_num_rows; j++){
            stand_mat[j][i] = (static_cast<double>(this->m_dataset[j][i]) - col_mean)/col_std;
        }
    }

    return stand_mat;

}