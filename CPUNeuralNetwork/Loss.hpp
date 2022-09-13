#ifndef CPU_LOSS
#define CPU_LOSS

#include "Vector.hpp"

namespace cpu{
    /** 
     * A class that contains two methods. The first method computes the loss
     * for the neural network using Binary Cross Entropy loss function while
     * the second method computes the loss using the derivative of the 
     * Binary Cross Entropy loss function. 
     * 
     * The two methods of this class naturally belong in the OutputLayer class.
     * However, if they belong to the OutputLayer class, any other class that needs 
     * to use the loss functionality would need an istance of the OutputLayer class
     * which makes the code messy. Thus, this Loss class was created in order to 
     * make access to the loss functionality not require an instance of the OutputLayer.
     * 
     */
    class Loss{

        public:

            Loss();

            double computeLoss(const cpu::Vector& a, const cpu::Vector& y);
            
            double computeLossPrime(const cpu::Vector& a, const cpu::Vector& y);
    };
}

#endif