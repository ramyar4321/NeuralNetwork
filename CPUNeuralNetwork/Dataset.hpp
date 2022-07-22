#ifndef CPU_Dataset
#define CPU_Dataset

#include <string>
#include <vector>
#include "Matrix.hpp"

namespace cpu{
    /**
     * Dataset used to repersent the dataset used to train
     * the neural network.
     */
    class Dataset{
        public:
            Dataset(int dataset_col_size, 
                    int dataset_row_size,
                    float split_size);

            void import_dataset(std::string &filename);

            Matrix X_train_split();
            Matrix X_test_split();
            std::vector<double> y_train_split();
            std::vector<double> y_test_split(); 

            Matrix get_dataet();



        private:
            // Store the CSV data into 2d vector
            Matrix m_dataset;
            // ratio = (train size)/((train size)+(test size))
            // Ratio will be used to split dataset into train test split
            float m_train_test_ratio;

    };
}

#endif // End of CPU_Dataset