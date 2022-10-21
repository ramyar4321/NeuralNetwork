#ifndef CPU_HIDDEN_LAYER
#define CPU_HIDDEN_LAYER

#include "../Matrices/Matrix.hpp"
#include "../Matrices/Vector.hpp"

namespace cpu{
    /**
     * HiddenLayer class to represent a hidden layer
     * of a neural network. The size of a this hidden layer
     * is specified during instantiation of this hidden layer class.
     * This hidden layer will not include a bias term.
    */
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
            void weightInitialization();
            void computeOutput(const cpu::Vector& a);
            void reluActivation();
            cpu::Vector forwardPropegation(const cpu::Vector& a);

            /*=======================*/
            // Methodes for backward propegation
            cpu::Vector reluPrime();
            void computeDelta(const cpu::Vector& W, const float& delta);
            void computeDelta(const cpu::Matrix& W, const cpu::Vector& delta);
            void computeGradient(const cpu::Vector& a);
            cpu::Vector backPropegation(const cpu::Vector& W, 
                                        const float& delta, const cpu::Vector& a);
            cpu::Vector backPropegation(const cpu::Matrix& W, 
                                        const cpu::Vector& delta, const cpu::Vector& a);

            /*=======================*/
            // Methodes for updating the weights
            void gradientDecent(const float& alpha);
            void updateWeigths(const float& alpha);

            /*=======================*/
            // Getter methods
            const cpu::Vector& a() const;
            const cpu::Matrix& W() const;
            const cpu::Matrix& dLdW() const;

            // Setter methods
            void W(const cpu::Matrix& W);
    };
}

#endif // End of CPU_HIDDEN_LAYER