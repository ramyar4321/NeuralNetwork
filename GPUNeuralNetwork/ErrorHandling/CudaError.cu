#include "CudaError.cuh"


/**
 * Check if CURAND API call was succesful.
 * If not successful, print where in the code 
 * the error occured and the error status.
*/
void gpu::CudaError::checkCurandError(curandStatus_t status, std::string error_message){

    if (status != CURAND_STATUS_SUCCESS) {
        std::cout << error_message << std::endl;
        switch (status) {
            case CURAND_STATUS_SUCCESS:
            std::cout << "CURAND_STATUS_SUCCESS" << std::endl;

            case CURAND_STATUS_VERSION_MISMATCH:
            std::cout << "CURAND_STATUS_VERSION_MISMATCH" << std::endl;

            case CURAND_STATUS_NOT_INITIALIZED:
            std::cout << "CURAND_STATUS_NOT_INITIALIZED" << std::endl;

            case CURAND_STATUS_ALLOCATION_FAILED:
            std::cout << "CURAND_STATUS_ALLOCATION_FAILED" << std::endl;

            case CURAND_STATUS_TYPE_ERROR:
            std::cout << "CURAND_STATUS_TYPE_ERROR" << std::endl;

            case CURAND_STATUS_OUT_OF_RANGE:
            std::cout << "CURAND_STATUS_OUT_OF_RANGE" << std::endl;

            case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
            std::cout << "CURAND_STATUS_LENGTH_NOT_MULTIPLE" << std::endl;

            case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            std::cout << "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED" << std::endl;

            case CURAND_STATUS_LAUNCH_FAILURE:
            std::cout << "CURAND_STATUS_LAUNCH_FAILURE" << std::endl;

            case CURAND_STATUS_PREEXISTING_FAILURE:
            std::cout << "CURAND_STATUS_PREEXISTING_FAILURE" << std::endl;

            case CURAND_STATUS_INITIALIZATION_FAILED:
            std::cout << "CURAND_STATUS_INITIALIZATION_FAILED" << std::endl;

            case CURAND_STATUS_ARCH_MISMATCH:
            std::cout << "CURAND_STATUS_ARCH_MISMATCH" << std::endl;

            case CURAND_STATUS_INTERNAL_ERROR:
            std::cout << "CURAND_STATUS_INTERNAL_ERROR" << std::endl;

            default:
                std:: cout << "Unknown CURAND error." << std::endl;
        }
        // Terminate process 
        exit(1);
    }
}