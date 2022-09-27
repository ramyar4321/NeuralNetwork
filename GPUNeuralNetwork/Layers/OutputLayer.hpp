#ifndef GPU_OUTPUT_LAYER
#define GPU_OUTPUT_LAYER

#include "../Matrix/Vector.hpp"

namespace gpu{
    class OutputLayer{
        private:

            /*=======================*/
            // private memeber variables

            float m_z;
            float m_a;
            float m_delta;
            gpu::Vector m_W;
            gpu::Vector m_dLdW;

        public:

            /*=======================*/
            // Constructor
            OutputLayer(int layerI_size);

            /*=======================*/
            // Vector operations
            void dot(float& z, const gpu::Vector& W, const gpu::Vector& s);
            void vecScalarMult(gpu::Vector& dLdW, const gpu::Vector& a, const float& delta);
            void vecScalarMultSub(gpu::Vector& W, const gpu::Vector& dLdW, const float& alpha);


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
            void WDeepCopy(const gpu::Vector& W);

    };
}

#endif // End of GPU_OUTPUT_LAYER