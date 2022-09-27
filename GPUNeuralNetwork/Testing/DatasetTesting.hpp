#ifndef DATASET_TESTING
#define DATASET_TESTING
#include "../Matrix/Matrix.hpp"
#include "../Matrix/Vector.hpp"

namespace gpu {
    /**
     * The purpose of this class is to test certain
     * methodes of the NeuralNetwork class. The tests will not be exhaustive.
     */
    class DatasetTesting {

        public:

        void test_import_dataset();
        void test_X_train_split();
        void test_X_test_split();
        void test_y_train_split();
        void test_y_test_split();

        void test_getColumn();
        void test_getRow();
        void test_setValue();

        void test_standardizeDataset();

        // Helper functions
        static bool areFloatEqual(float a, float b);
    

    };
}

#endif // End of DATASET_TESTING