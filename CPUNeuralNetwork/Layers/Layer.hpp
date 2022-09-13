#ifndef CPU_LAYER
#define CPU_LAYER

#include "../Vector.hpp"
#include "../Matrix.hpp"

namespace cpu{
    /**
     * 
     * An Abstract class for a given layer of neural network.
     * Any given layer should have a forward propagation, 
     * backward propagation ,and update weights methods.
     * 
     * The purpose of this class is to allow a vector of layers 
     * to be defined in the Neural Network class.
     * 
     */
    class Layer{
        public:
        
            // Layer methods for forward propegation.
            virtual void weightInitialization() = 0;
            virtual cpu::Vector forwardPropegation(const cpu::Vector& a) = 0;

            // Layer methods for backward propegation.
            // Backward propegation methode for the output layer
            virtual cpu::Vector backPropegation(const cpu::Vector& y, const cpu::Vector& a);
            // Backward propegation methode for the hidden layer
            virtual cpu::Vector backPropegation(const cpu::Matrix& W, 
                                                const cpu::Vector& delta, 
                                                const cpu::Vector& a);


            // Layer methods for gradient descent.
            virtual void updateWeigths(const double& alpha) = 0;

            // Getter methods
            virtual const cpu::Vector& a() const = 0;
            virtual const cpu::Matrix& W() const = 0;

            //Setter methods
            virtual void W(const cpu::Matrix& W) = 0;

    };
}

#endif // End of CPU_LAYER