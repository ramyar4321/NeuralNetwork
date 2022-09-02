#ifndef CPU_OUTPUT_LAYER
#define CPU_OUTPUT_LAYER

#include "Vector.hpp"

namespace cpu{
    class OutputLayer{
        private:
            float m_z;
            float m_a;
            float m_delta;

            cpu::Vector m_W;
            cpu::Vector m_dLdW;

        public:

            /*=======================*/
            // Constructor
            OutputLayer(int prev_layer_size);


            /*=======================*/
            // Methodes for forward propegation
            void computeOutput(const cpu::Vector& a);
            void sigmoidActivation();
            void forwardPropegation(const cpu::Vector& a);
            double bceLoss(const double &y);

            /*=======================*/
            // Methodes for backward propegation
            double sigmoidPrime();
            double bceLossPrime(const double &y);
            void computeDelta(const double& y);
            void computeGradient(const cpu::Vector& a);
            void backPropegation(const double& y, const cpu::Vector& a);

            /*=======================*/
            // Methodes for updating the weights
            void gradientDecent(const double& alpha);
            void updateWeigths(const double& alpha);

    };
}

#endif // End of CPU_OUTPUT_LAYER