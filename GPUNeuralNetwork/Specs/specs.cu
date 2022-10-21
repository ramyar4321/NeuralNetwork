#include "specs.cuh"
#include <iostream>

namespace gpu{
    /**
     * Programatically print specifications of GPU.
     * 
    */
    void getGPUSpecs(){

    const int kb = 1024;
    const int mb = kb * kb;

    // Get info about CUDA version, driver and runtime
    std::cout << "CUDA version:   v" << CUDART_VERSION << std::endl;
    int driver;
    int runtime;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    std::cout << "Driver: " << driver << " Runtime: " << runtime << std::endl << std::endl;    

    int d_count;
    cudaGetDeviceCount(&d_count);
    std::cout << "This PC has " << d_count << " GPU(s)" << std::endl << std::endl;

    for(int i = 0; i < d_count; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);
        // Get device name and capabilities
        std::cout << "GPU # " << i << " : " << props.name 
                  << ": " << props.major << "." << props.minor << std::endl;

        // Get information about device memeory
        std::cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << std::endl;
        std::cout << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;

        // Get information about threads
        std::cout << "  Warp size:         " << props.warpSize << std::endl;
        std::cout << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
        std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
        std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << std::endl;
        std::cout << std::endl;
        // The number of SMs
        std::cout << "Number of SMs: " << props.multiProcessorCount << std::endl;
    }
    }
}