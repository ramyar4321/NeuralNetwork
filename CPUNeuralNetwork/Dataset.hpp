#ifndef CPU_Dataset
#define CPU_Dataset

#include <string>
#include <vector>

namespace cpu{
    /**
     * Dataset used to repersent the dataset used to train
     * the neural network.
     */
    class Dataset{
        public:
            Dataset(unsigned int dataset_col_size, 
                    unsigned int dataset_row_size,
                    float split_size);

            void import_dataset(std::string &filename);

            std::vector<std::vector<double> > X_train_split();
            std::vector<std::vector<double> > X_test_split();
            std::vector<double> y_train_split();
            std::vector<double> y_test_split(); 

            std::vector<std::vector<double> > get_dataet();



        private:
            // Store the CSV data into 2d vector
            std::vector<std::vector<double> > m_dataset;
            // ratio = (train size)/((train size)+(test size))
            // Ratio will be used to split dataset into train test split
            float m_train_test_ratio;

    };
}

#endif // End of CPU_Dataset