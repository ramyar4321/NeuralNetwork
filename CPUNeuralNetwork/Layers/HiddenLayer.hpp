#ifndef CPU_HIDDEN_LAYER
#define CPU_HIDDEN_LAYER

#include "Layer.hpp"
#include "../Matrix.hpp"
#include "../Vector.hpp"

namespace cpu{
    class HiddenLayer: public cpu::Layer{
       private:
            cpu::Vector m_z;

        public:
            cpu::Vector m_a;
            cpu::Vector m_delta;
            cpu::Matrix m_W;
            cpu::Matrix m_dLdW;


            /*=======================*/
            // Constructor
            HiddenLayer(int layerI_size, int layerJ_size);

            /*=======================*/
            // Methodes for forward propegation
            void weightInitialization();
            void computeOutput(const cpu::Vector& a);
            void computeActivation();
            void forwardPropegation(const cpu::Vector& a);

            /*=======================*/
            // Methodes for backward propegation
            cpu::Vector computeActivationPrime();
            void computeDelta(const cpu::Vector& W, const double& delta);
            void computeDelta(const cpu::Matrix& W, const cpu::Vector& delta_);
            void computeGradient(const cpu::Vector& a);
            void backPropegation(const cpu::Vector& W, const double& delta, const cpu::Vector& a);
            void backPropegation(const cpu::Matrix& W, const cpu::Vector& delta, const cpu::Vector& a);

            /*=======================*/
            // Methodes for updating the weights
            void gradientDecent(const double& alpha);
            void updateWeigths(const double& alpha);
    };
}

#endif // End of CPU_HIDDEN_LAYER