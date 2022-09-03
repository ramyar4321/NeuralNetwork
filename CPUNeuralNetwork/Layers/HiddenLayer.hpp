#ifndef CPU_HIDDEN_LAYER
#define CPU_HIDDEN_LAYER

#include "Matrix.hpp"
#include "Vector.hpp"

namespace cpu{
    class HiddenLayer{
       private:
            cpu::Vector m_z;
            cpu::Vector m_a;
            cpu::Vector m_delta;

            cpu::Matrix m_W;
            cpu::Matrix m_dLdW;

        public:

            /*=======================*/
            // Constructor
            HiddenLayer(int layerI_size, int layerJ_size);

            /*=======================*/
            // Methodes for forward propegation
            void computeOutput(const cpu::Vector& a);
            void reluActivation();
            void forwardPropegation(const cpu::Vector& a);

            /*=======================*/
            // Methodes for backward propegation
            cpu::Vector reluPrime();
            void computeDelta(const cpu::Vector& W, const double& delta);
            void computeDelta(const cpu::Matrix& W, const cpu::Vector& delta);
            void computeGradient(const cpu::Vector& a);
            void backPropegation(const cpu::Vector& W, const double& delta, const cpu::Vector& a);
            void backPropegation(const cpu::Matrix& W, const cpu::Vector& delta, const cpu::Vector& a);
    };
}

#endif // End of CPU_HIDDEN_LAYER