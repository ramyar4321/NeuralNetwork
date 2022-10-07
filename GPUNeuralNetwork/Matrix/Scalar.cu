#include "Scalar.cuh"

gpu::Scalar::Scalar(float init_val){
    this->allocateMemHost(init_val);
    this->allocateMemDevice();
    // copy init_val to device
    this->copyHostToDevice();
}

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