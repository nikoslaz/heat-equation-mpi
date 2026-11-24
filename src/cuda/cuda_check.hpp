#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Abort with file and line if a CUDA runtime call fails.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                         cudaGetErrorString(err_), __FILE__, __LINE__);         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)
