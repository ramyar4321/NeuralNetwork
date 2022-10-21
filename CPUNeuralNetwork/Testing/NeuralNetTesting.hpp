#ifndef CPU_TESTING
#define CPU_TESTING
#include "../Matrices/Matrix.hpp"
#include "../Matrices/Vector.hpp"

namespace cpu {
    /**
     * The purpose of this class is to test certain
     * methodes of the NeuralNetwork class. The tests will not be exhaustive.
     */
    class NeuralNetTesting {

        public:

        // Test methodes of the NeuralNetwork class
        void test_forwardPropegation();
        void test_backPropegation();
        void test_gradientDescent();
     

        // Helper functions

        static bool areFloatEqual(float a, float b);
        float computeQuadraticLoss(cpu::Vector& w);
        cpu::Matrix computeGradientQuadraticLoss(cpu::Vector& w);

    };
}

#endif // End of CPU_TESTING