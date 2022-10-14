#include "Scalar.cuh"

gpu::Scalar::Scalar(float init_val){
    this->allocateMemHost(init_val);
    this->allocateMemDevice();
    // copy init_val to device
    this->copyHostToDevice();
}

gpu::Scalar::Scalar( gpu::Scalar& other):
                 h_scalar(other.h_scalar),
                 d_scalar(other.d_scalar)       
{}

void gpu::Scalar::allocateMemHost(float init_val){
    this->h_scalar = std::make_shared<float>(init_val);
}
void gpu::Scalar::allocateMemDevice(){
    this->d_scalar = std::shared_ptr<float>(nullptr,  [&](float* ptr){ cudaFree(ptr);});
    cudaMalloc((void**) &this->d_scalar, sizeof(float));
}
void gpu::Scalar::copyHostToDevice(){
    cudaMemcpy(this->d_scalar.get(), this->h_scalar.get(), sizeof(float), cudaMemcpyHostToDevice);
}
void gpu::Scalar::copyDeviceToHost(){
    cudaMemcpy(this->h_scalar.get(), this->d_scalar.get(), sizeof(float), cudaMemcpyDeviceToHost);
}

gpu::Scalar& gpu::Scalar::operator=(const gpu::Scalar& rhs){
    // Check if object is being assigned to itself.
    if(this == &rhs){
        return *this;
    }

    this->h_scalar = rhs.h_scalar;
    this->d_scalar = rhs.d_scalar;

    return *this;
}

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