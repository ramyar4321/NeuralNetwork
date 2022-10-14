#ifndef GPU_SCALAR
#define GPU_SCALAR
#include <memory>

namespace gpu{
    class Scalar{
        public:

            std::shared_ptr<float> h_scalar;
            std::shared_ptr<float> d_scalar;

            Scalar(float init_val);
            Scalar( Scalar& other);


            void allocateMemHost(float init_val);
            void allocateMemDevice();
            void copyHostToDevice();
            void copyDeviceToHost();

            Scalar& operator=(const Scalar& rhs);
            bool operator==(Scalar& rhs);
            
    };
}

#endif // End of GPU_SCALAR