#ifndef GPU_TESTING
#define GPU_TESTING
#include "../Matrix/Matrix.cuh"
#include "../Matrix/Vector.cuh"

namespace gpu {
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
        float computeQuadraticLoss(gpu::Vector& w);
        gpu::Matrix computeGradientQuadraticLoss(gpu::Vector& w);

    };
}

#endif // End of GPU_TESTING