#ifndef GPU_CUDA_ERROR
#define GPU_CUDA_ERROR

#include <string>
#include <iostream>
#include <curand.h>

namespace gpu{
    /**
     * The purpose of this class it to check for errors thrown 
     * by CUDA and CURAND API calls.
     *  
    */
    class CudaError {
        public:

            static void checkCurandError(curandStatus_t status, std::string err_msg);

            template<typename T>
            static void checkCudaError(T cudaAPICall, const std::string& error_message);

    };
}

/**
 * Check is CUDA API call produced an error.
 * If so, print where the error occured in the code
 * and the actual error.
*/
template<typename T>
void gpu::CudaError::checkCudaError(T cudaAPICall, const std::string& error_message){

    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        std::cout << error_message << std::endl;
        std::cout << cudaGetErrorString(err) << std::endl;
        // Terminate process to reset device.
        exit(1);
    }
}

#endif // End of GPU_CUDA_ERROR