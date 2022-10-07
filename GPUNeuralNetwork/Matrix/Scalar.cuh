#include <memory>

namespace gpu{
    class Scalar{
        public:

            std::shared_ptr<float> h_scalar;
            std::shared_ptr<float> d_scalar;

            Scalar(float init_val);

            void allocateMemHost(float init_val);
            void allocateMemDevice();
            void copyHostToDevice();
            void copyDeviceToHost();
            
    };
}