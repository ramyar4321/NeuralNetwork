#include "Scalar.cuh"
#include "../ErrorHandling/CudaError.cuh"

//================================//
// Constructors.
//================================//

/**
 * Constructor for Scalar class.
 * Constructor scalar using float value. 
*/
gpu::Scalar::Scalar(float init_val){
    this->allocateMemHost(init_val);
    this->allocateMemDevice();
    // copy init_val to device
    this->copyHostToDevice();
}

/**
 * Copy Constructor.
 * 
*/
gpu::Scalar::Scalar( gpu::Scalar& other):
                 h_scalar(other.h_scalar),
                 d_scalar(other.d_scalar)       
{}

//================================//
// Memeory management
//================================//

/**
 * Allocate scalar value on host.
 * Initialize scalar to init_val.
*/
void gpu::Scalar::allocateMemHost(float init_val){
    this->h_scalar = std::make_shared<float>(init_val);
}

/**
 * Allocate space on device for scalar value.
*/
void gpu::Scalar::allocateMemDevice(){

    std::string cudaFree_err_msg = "cudaFree failed in Scalar allocateMemDevice.";
    std::string cudaMalloc_err_msg = "cudaMalloc failed in Scalar allocateMemDevice";

    this->d_scalar = std::shared_ptr<float>(nullptr,  
                                            [&](float* ptr){ gpu::CudaError::checkCudaError(cudaFree(ptr), cudaFree_err_msg);});
    gpu::CudaError::checkCudaError(cudaMalloc((void**) &this->d_scalar, sizeof(float)), cudaMalloc_err_msg);
}

/**
 * Copy scalar value from host to device.
 * 
*/
void gpu::Scalar::copyHostToDevice(){
    std::string cudaMemcpy_err_msg = "cudaMemcpy failed in Scalar copyHostToDevice";
    gpu::CudaError::checkCudaError(cudaMemcpy(this->d_scalar.get(), this->h_scalar.get(), 
                                                sizeof(float), cudaMemcpyHostToDevice), cudaMemcpy_err_msg);
}

/**
 * Copy scalar value from device to host.
*/
void gpu::Scalar::copyDeviceToHost(){
    std::string cudaMemcpy_err_msg = "cudaMemcpy failed in Scalar copyDeviceToHost";
    gpu::CudaError::checkCudaError(cudaMemcpy(this->h_scalar.get(), this->d_scalar.get(), 
                                                sizeof(float), cudaMemcpyDeviceToHost), cudaMemcpy_err_msg);
}

//================================//
// Operators.
//================================//

/**
 * Overload assignment operator.
*/
gpu::Scalar& gpu::Scalar::operator=(const gpu::Scalar& rhs){
    // Check if object is being assigned to itself.
    if(this == &rhs){
        return *this;
    }

    this->h_scalar = rhs.h_scalar;
    this->d_scalar = rhs.d_scalar;

    return *this;
}

/**
 * Overload equality operator in order to 
 * allow comparison between two scalar objects.
*/
bool gpu::Scalar::operator==(gpu::Scalar& rhs){
    bool areEqual = true;

    // Fixed error for comparison between two given values
    constexpr double epsilon = 0.01; 

    rhs.copyDeviceToHost();
    this->copyDeviceToHost();

    // Variables to store scalar values to be compared
    float this_val = *this->h_scalar.get();
    float rhs_val = *rhs.h_scalar.get();

    if(!(std::abs(this_val - rhs_val) < epsilon)){
        areEqual = false;
    }

    return areEqual;
}