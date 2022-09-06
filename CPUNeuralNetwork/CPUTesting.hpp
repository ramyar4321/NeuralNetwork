#ifndef CPU_TESTING
#define CPU_TESTING
#include "Matrix.hpp"
#include "Vector.hpp"

namespace cpu {
    /**
     * The purpose of this class is to test certain
     * methodes of the NeuralNetwork class. The tests will not be exhaustive.
     */
    class Testing {

        public:

        // Test methodes of the NeuralNetwork class
        void test_forwardPropegation();
        void test_backPropegation();
        void test_gradientDescent();

        // Test methodes of the Dataset class
        void test_import_dataset();
        void test_X_train_split();
        void test_X_test_split();
        void test_y_train_split();
        void test_y_test_split();

        void test_getColumn();

        void test_setValue();




        // Tests methodes for Matrix class
        void test_computeMean();
        void test_computeStd();
        void test_standardizeMatrix();
        void test_getRow();
    
        

        // Helper functions

        static bool areFloatEqual(double a, double b);
        double computeQuadraticLoss(cpu::Vector& w);
        cpu::Matrix computeGradientQuadraticLoss(cpu::Vector& w);

    };
}

#endif // End of CPU_TESTING