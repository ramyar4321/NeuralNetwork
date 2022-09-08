#ifndef CPU_LAYER
#define CPU_LAYER

#include "../Vector.hpp"

namespace cpu{
    /**
     * 
     * An Abstract class for a given layer of neural network.
     * Any given layer hsould have a forward propegation, 
     * backward propegation ,and update weights methodes.
     * 
     * The purpose of this class is to allow a vector of layers 
     * to be defined in the Neural Network class.
     * 
     */
    class Layer{
        public:

            // Layer methodes for forward propegation.
            virtual void weightInitialization() = 0;
            virtual void computeOutput(const cpu::Vector& a) = 0;
            virtual void computeActivation() = 0;
            virtual void forwardPropegation(const cpu::Vector& a) = 0;

            // Layer methodes for backward propegation.
            virtual void computeGradient(const cpu::Vector& a) = 0;

            // Layer methodes for gradient descent.
            virtual void gradientDecent(const double& alpha) = 0;
            virtual void updateWeigths(const double& alpha) = 0;

    };
}

#endif // End of CPU_LAYER