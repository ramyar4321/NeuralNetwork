#ifndef CPU_OUTPUT_LAYER
#define CPU_OUTPUT_LAYER

#include "../Matrices/Vector.hpp"

namespace cpu{
    /**
     * OutputLayer class to represent an output layer 
     * of a neural network. The output layer will contain
     * only one output neuron.
    */
    class OutputLayer{
        private:

            /*=======================*/
            // private memeber variables

            float m_z;
            float m_a;
            float m_delta;
            cpu::Vector m_W;
            cpu::Vector m_dLdW;

        public:

            /*=======================*/
            // Constructor
            OutputLayer(int layerI_size);


            /*=======================*/
            // Methodes for forward propegation
            void weightInitialization();
            void computeOutput(const cpu::Vector& a);
            void sigmoidActivation();
            float forwardPropegation(const cpu::Vector& a);
            float bceLoss(const float &y);

            /*=======================*/
            // Methodes for backward propegation
            float sigmoidPrime();
            float bceLossPrime(const float &y);
            void computeDelta(const float& y);
            void computeGradient(const cpu::Vector& a);
            float backPropegation(const float& y, const cpu::Vector& a);

            /*=======================*/
            // Methodes for updating the weights
            void gradientDecent(const float& alpha);
            void updateWeigths(const float& alpha);

            /*=======================*/
            // Setter and getter methods

            // Getter methods
            const float& a() const;
            const cpu::Vector& W() const;
            const cpu::Vector& dLdW() const;

            // Setter methods
            void W(const cpu::Vector& W);

    };
}

#endif // End of CPU_OUTPUT_LAYER