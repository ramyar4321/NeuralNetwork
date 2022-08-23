#ifndef GPU_Dataset
#define GPU_Dataset

#include <string>
#include <vector>

namespace gpu{
    /**
     * Dataset used to represent the dataset used to train
     * the neural network.
     */
    class Dataset{
        public:
            Dataset(int num_rows, int num_cols, float split_size);
            Dataset(int num_rows, int num_cols);

            void import_dataset(std::string &filename);

            const std::vector<float>& operator[](const int &input) const;
            std::vector<float>& operator[](const int &input);

            Dataset X_train_split();
            Dataset X_test_split();
            std::vector<float> y_train_split();
            std::vector<float> y_test_split(); 

            void setValues(std::vector<float>& y);

            std::vector<std::vector<float> > get_dataset();

            //Helper fucntion for train test split
            Dataset getSubDataset(int& start_ri, int& end_ri,
                                 int& start_ci, int& end_ci);
            std::vector<float> getCol(int& ci, int& start_ri, int& end_ri);
            std::vector<float> getCol(int& ci);
            std::vector<float> getRow(int& ri);

            int get_num_rows() const;
            int get_num_cols() const;

            float computeMean(int& ci);
            float computeStd(int& ci);
            float computeStd(int& ci, float &mean);

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
            std::vector<std::vector<float> > m_dataset;
            // Ratio will be used to split dataset into train test split
            float m_train_test_ratio;

    };
}

#endif // End of CPU_Dataset