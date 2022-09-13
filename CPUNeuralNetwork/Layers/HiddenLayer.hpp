#ifndef CPU_HIDDEN_LAYER
#define CPU_HIDDEN_LAYER

#include "Layer.hpp"
#include "../Matrix.hpp"
#include "../Vector.hpp"

namespace cpu{
    class HiddenLayer: public cpu::Layer{
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
            void weightInitialization() override;
            void computeOutput(const cpu::Vector& a);
            void computeActivation();
            cpu::Vector forwardPropegation(const cpu::Vector& a) override;

            /*=======================*/
            // Methodes for backward propegation
            cpu::Vector computeActivationPrime();
            void computeDelta(const cpu::Matrix& W, const cpu::Vector& delta);
            void computeGradient(const cpu::Vector& a);
            cpu::Vector backPropegation(const cpu::Matrix& W, 
                                        const cpu::Vector& delta, 
                                        const cpu::Vector& a) override;

            /*=======================*/
            // Methodes for updating the weights
            void gradientDecent(const double& alpha);
            void updateWeigths(const double& alpha);

            // Getter methods
            virtual const cpu::Vector& a() const override;
            virtual const cpu::Matrix& W() const override;

            //Setter methods
            void W(const cpu::Matrix& W) override;
    };
}

#endif // End of CPU_HIDDEN_LAYER