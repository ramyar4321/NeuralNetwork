#ifndef GPU_HIDDEN_LAYER
#define GPU_HIDDEN_LAYER

#include "../Matrix/Matrix.cuh"
#include "../Matrix/Vector.cuh"

namespace gpu{
    class HiddenLayer{
       private:
            gpu::Vector m_z;
            gpu::Vector m_a;
            gpu::Vector m_delta;
            gpu::Matrix m_W;
            gpu::Matrix m_dLdW;

        public:


            /*=======================*/
            // Constructor
            HiddenLayer(int layerI_size, int layerJ_size);

            /*=======================*/
            // Matrix operations

            //void matrixVectorMult(gpu::Vector& z, const gpu::Matrix& W, const gpu::Vector& a);
            //void matrixTransposeVectorMult(gpu::Vector& delta, const gpu::Matrix& W, 
            //                               const gpu::Vector& delta_, const gpu::Vector& z);
            //void matrixVectorMult(gpu::Vector& delta, const gpu::Vector& W, 
            //                      const float& delta_, const gpu::Vector& z);
            //void tensor(gpu::Matrix& W, const gpu::Vector& a, const gpu::Vector& delta);
            //void matrixScalarMultSub(gpu::Matrix& W, const gpu::Matrix& dLdW, const float& alpha);

            /*=======================*/
            // Methodes for forward propegation
            void weightInitialization();
            void computeOutput(const gpu::Vector& a);
            void reluActivation();
            gpu::Vector forwardPropegation(const gpu::Vector& a);

            /*=======================*/
            // Methodes for backward propegation
            //float reluPrime(const float& z);
            //gpu::Vector reluPrime();
            void computeDelta(const gpu::Vector& W, const float& delta);
            void computeDelta(const gpu::Matrix& W, const gpu::Vector& delta);
            void computeGradient(const gpu::Vector& a);
            gpu::Vector backPropegation(const gpu::Vector& W, 
                                        const float& delta, const gpu::Vector& a);
            gpu::Vector backPropegation(const gpu::Matrix& W, 
                                        const gpu::Vector& delta, const gpu::Vector& a);

            /*=======================*/
            // Methodes for updating the weights
            void gradientDecent(const float& alpha);
            void updateWeigths(const float& alpha);

            /*=======================*/
            // Getter methods
            const gpu::Vector& a() const;
            const gpu::Matrix& W() const;
            const gpu::Matrix& dLdW() const;

            // Setter methods
            void W(const gpu::Matrix& W);
            void WDeepCopy(gpu::Matrix& W);
    };
}

#endif // End of GPU_HIDDEN_LAYER