// GPU kernels for the 2D heat equation, callable from plain C++ (implemented in heat_gpu.cu).
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

// Kernels that advance the grid by one time step.
enum class StepKernel {
    Naive,    // one thread per point, every value loaded from global memory (through the caches)
    Shared,   // each block first copies its tile plus a one-point halo into shared memory
};

// Largest number of time steps the temporal kernel does in one launch.
constexpr int kMaxTemporalSteps = 8;

// One time step on rows [rowBegin, rowEnd), columns 1..N-2, of an array with N columns:
//   uNew = u + alpha * (u_up + u_down + u_left + u_right - 4 u)
// Rows rowBegin-1 and rowEnd must exist (boundary or ghost rows); they are read, not written.
template <typename T>
void launch_step_rows(const T* u, T* uNew, int N, int rowBegin, int rowEnd, T alpha,
                      StepKernel kernel = StepKernel::Naive, cudaStream_t stream = 0);

// S time steps (1 <= S <= kMaxTemporalSteps) on the whole N x N grid in a single launch
// (temporal blocking): each block keeps its tile in shared memory for all S steps, so the
// grid is read and written once per S steps instead of once per step.
template <typename T>
void launch_temporal(const T* u, T* uNew, int N, int S, T alpha, cudaStream_t stream = 0);

// Sum of n values on the GPU, accumulated in double. Blocks until the result is ready.
template <typename T>
double gpu_sum(const T* x, size_t n, cudaStream_t stream = 0);
