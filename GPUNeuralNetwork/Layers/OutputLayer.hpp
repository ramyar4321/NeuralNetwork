#ifndef GPU_OUTPUT_LAYER
#define GPU_OUTPUT_LAYER

#include "../Matrix/Vector.cuh"
#include "../Matrix/Scalar.cuh"

namespace gpu{
    /**
     * OutputLayer class to represent an output layer 
     * of a neural network. The output layer will contain
     * only one output neuron.
    */
    class OutputLayer{
        private:

            /*=======================*/
            // private memeber variables

            gpu::Scalar m_z;
            float m_a;
            float m_delta;
            gpu::Vector m_W;
            gpu::Vector m_dLdW;

        public:

            /*=======================*/
            // Constructor
            OutputLayer(int layerI_size);

            /*=======================*/
            // Methodes for forward propegation

            void weightInitialization();
            void computeOutput(const gpu::Vector& a);
            void sigmoidActivation();
            float forwardPropegation(const gpu::Vector& a);
            float bceLoss(const float &y);

            /*=======================*/
            // Methodes for backward propegation
            float sigmoidPrime();
            float bceLossPrime(const float &y);
            void computeDelta(const float& y);
            void computeGradient(const gpu::Vector& a);
            float backPropegation(const float& y, const gpu::Vector& a);

            /*=======================*/
            // Methodes for updating the weights
            void gradientDecent(const float& alpha);
            void updateWeigths(const float& alpha);

            /*=======================*/
            // Setter and getter methods

            // Getter methods
            const float& a() const;
            const gpu::Vector& W() const;
            const gpu::Vector& dLdW() const;

            // Setter methods
            void W(const gpu::Vector& W);
            void WDeepCopy(gpu::Vector& W);

    };
}

#endif // End of GPU_OUTPUT_LAYER