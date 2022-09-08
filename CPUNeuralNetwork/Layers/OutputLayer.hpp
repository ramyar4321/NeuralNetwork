#ifndef CPU_OUTPUT_LAYER
#define CPU_OUTPUT_LAYER

#include "Layer.hpp"
#include "../Vector.hpp"

namespace cpu{
    class OutputLayer: public cpu::Layer{
        private:

            /*=======================*/
            // private memeber variables

            double m_z;

        public:


            /*=======================*/
            // public memeber variables
            double m_a;
            double m_delta;
            cpu::Vector m_W;
            cpu::Vector m_dLdW;

            /*=======================*/
            // Constructor
            OutputLayer(int layerI_size);


            /*=======================*/
            // Methodes for forward propegation
            void weightInitialization() override;
            void computeOutput(const cpu::Vector& a) override;
            void computeActivation() override;
            void forwardPropegation(const cpu::Vector& a) override;
            double computeLoss(const double &y);

            /*=======================*/
            // Methodes for backward propegation
            double computeActivationPrime();
            double computeLossPrime(const double &y);
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