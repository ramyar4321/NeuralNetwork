#ifndef CPU_TESTING
#define CPU_TESTING

namespace cpu {
    /**
     * The purpose of this class is to test certain
     * methodes of the NeuralNetwork class. The tests will not be exhaustive.
     */
    class Testing {

        public:

        void test_compute_outputs();
        void test_relu_activation();
        void test_sigmoid_activation();
        void test_compute_loss();

        // Helper functions

        static bool areFloatEqual(float a, float b);

    };
}

#endif // End of CPU_TESTING