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
            Dataset(int num_rows, int num_cols, float split_size);
            Dataset(int num_rows, int num_cols);
            Dataset(std::initializer_list< std::initializer_list<double> > ilist);

            void import_dataset(std::string &filename);

            const std::vector<double>& operator[](const int &input) const;
            std::vector<double>& operator[](const int &input);
            bool operator==(const cpu::Dataset& rhs) const;

            Dataset X_train_split();
            Dataset X_test_split();
            std::vector<double> y_train_split();
            std::vector<double> y_test_split(); 

            void setValues(std::vector<double>& y);

            std::vector<std::vector<double> > get_dataset();

            //Helper fucntion for train test split
            Dataset getSubDataset(int& start_ri, int& end_ri,
                                 int& start_ci, int& end_ci);
            std::vector<double> getCol(int& ci, int& start_ri, int& end_ri);
            std::vector<double> getCol(int& ci);
            std::vector<double> getRow(int& ri);

            double get_num_rows() const;
            double get_num_cols() const;

            double computeMean(int& ci);
            double computeStd(int& ci);
            double computeStd(int& ci, double &mean);

            Dataset standardizeMatrix();



        private:
            // Store the CSV data into 2d array
            // The dataset we will be using Haberman
            // cancer survival dataset with 306 rows and
            // 4 columns.
            //double m_dataset[306][4];
            //static constexpr int m_num_rows = 306;
            //static constexpr int m_num_cols = 4;
            //std::array<std::array<double, m_num_cols>, m_num_rows> m_dataset;
            // ratio = (train size)/((train size)+(test size))
            int m_num_rows;
            int m_num_cols;
            std::vector<std::vector<double> > m_dataset;
            // Ratio will be used to split dataset into train test split
            float m_train_test_ratio;

    };
}

#endif // End of CPU_Dataset