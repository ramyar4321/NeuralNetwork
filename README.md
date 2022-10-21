# CUDA Neural Network

There were three pruposes for this project. The first was to explore CUDA programming.
The second was to create a neural network without the uses of any major libraries. 
The last reason was to get familiar with C++ programming. 

# Software System requirements

For the CPU portion of this project, you will need a C++ compiler.
I used visual studio code with g++ compiler.
 You can follow the instructions [here](https://code.visualstudio.com/docs/languages/cpp)
 to setup the g++ compiler.

However, for the GPU portion of this project, you will need to have the [CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) and a [Nvidia GPU](https://www.nvidia.com/en-us/geforce/graphics-cards/).

Make sure to download the CUDA version compatible with your GPU capabilty.

If you have a Nvidia GPU device, you can run [specs.cu file](\GPUNeuralNetwork\Specs\specs.cu) by
uncommenting out `gpu::getGPUSpecs();` in the [main.cu file](\GPUNeuralNetwork\main.cu). The 
GPU device specifications for my system is the following.
```
CUDA version:   v10000
Driver: 11070 Runtime: 10000

This PC has 1 GPU(s)

GPU # 0 : NVIDIA GeForce GTX 960M: 5.0
  Global memory:   2047mb
  Block registers: 65536

  Warp size:         32
  Threads per block: 1024
  Max block dimensions: [ 1024, 1024, 64 ]
  Max grid dimensions:  [ 2147483647, 65535, 65535 ]

Number of SMs: 5
```



After installing the CUDA toolkit, you will have access to [nvcc](https://en.wikipedia.org/wiki/Nvidia_CUDA_Compiler). However, if you want to use Micsoft Visual Studios, make sure
to have it installed before installing the CUDA toolkit. Furthermore, make sure 
the Microsoft Visual Studio version is compatible with the CUDA version you installed.

 Run `nvcc --version` from the command prompt to get information about 
the nvcc version you have installed. I have the following version installed. 
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:04_Central_Daylight_Time_2018
Cuda compilation tools, release 10.0, V10.0.130
```

CUDA does not support g++ on windows. You will need `cl.exe` that comes with 2015 Microsoft Visual Studios.
If you do not have 2015 Microsoft Visual Studios installed, install the 2015 Micosoft Visual C++ Build Tools.
Located the path to the `cl.exe` file and add it to eviroment variables.