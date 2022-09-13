#ifndef CPU_OUTPUT_LAYER
#define CPU_OUTPUT_LAYER

#include "Layer.hpp"
#include "../Loss.hpp"
#include "../Matrix.hpp"
#include "../Vector.hpp"

namespace cpu{
    class OutputLayer: public cpu::Layer{
        private:

            cpu::Loss bceLoss;

            cpu::Vector m_z;
            cpu::Vector m_a;
            cpu::Vector m_delta;
            cpu::Matrix m_W;
            cpu::Matrix m_dLdW;

        public:

            /*=======================*/
            // Constructor
            OutputLayer(int layerI_size, int layerJ_size);
            

            /*=======================*/
            // Methods for forward propagation
            void weightInitialization() override;
            void computeOutput(const cpu::Vector& a);
            void computeActivation();
            cpu::Vector forwardPropegation(const cpu::Vector& a) override;
            //double computeLoss(const cpu::Vector &y);

            /*=======================*/
            // Methods for backward propagation
            cpu::Vector computeActivationPrime();
            //double computeLossPrime(const cpu::Vector& y);
            void computeDelta(const cpu::Vector& y);
            void computeGradient(const cpu::Vector& a);
            cpu::Vector backPropegation(const cpu::Vector& y, const cpu::Vector& a) override;

            /*=======================*/
            // Methodes for updating the weights
            void gradientDecent(const double& alpha);
            void updateWeigths(const double& alpha) override;

            // Getter Methods
            const cpu::Vector& a() const override;
            const cpu::Matrix& W() const override;

            //Setter methods
            void W(const cpu::Matrix& W) override;

    };
}

#endif // End of CPU_OUTPUT_LAYER