#ifndef CPU_OUTPUT_LAYER
#define CPU_OUTPUT_LAYER

#include "Vector.hpp"

namespace cpu{
    class OutputLayer{
        private:

            /*=======================*/
            // private memeber variables

            double m_z;
            //double m_y;

            cpu::Vector m_dLdW;

        public:


            /*=======================*/
            // public memeber variables
            double m_a;
            double m_delta;
            cpu::Vector m_W;

            /*=======================*/
            // Constructor
            OutputLayer(int layerI_size);


            /*=======================*/
            // Methodes for forward propegation
            void weightInitialization();
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