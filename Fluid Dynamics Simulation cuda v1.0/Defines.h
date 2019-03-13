#include "cuda_runtime.h"
#include <iostream>
#include <cassert>

#pragma once

#ifndef DEFINES
#define DEFINES

#define RES 600										//resolution of the window
#define ALENGTH (RES+2)*(RES+2)						//total array size

#define VISC 1.0f									//viscosity constant
#define DT 0.04f									//static delta time

#define TPBLOCK 256									//threads per block
#define NBLOCK (ALENGTH + TPBLOCK - 1) / TPBLOCK	//number of blocks

#define CUDA_CHECK(fn) {\
                const cudaError_t rc = (fn);\
                if (rc != cudaSuccess) {\
                                std::cout << "CUDA Error: " << cudaGetErrorString(rc) << " (" << rc << ")" << std::endl;\
                                cudaDeviceReset();\
                                assert(0);\
                }\
}

#define CUDA_CHECK_POST() {\
                const cudaError_t rc = cudaGetLastError();\
                if (rc != cudaSuccess) {\
                                std::cout << "CUDA Error: " << cudaGetErrorString(rc) << " (" << rc << ")" << std::endl;\
                                cudaDeviceReset();\
                                assert(0);\
                }\
}

#endif