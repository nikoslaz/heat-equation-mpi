// The CUDA kernels behind a plain C++ interface (implemented in kernels.cu), so the programs that use them
// compile with an ordinary C++ compiler and only kernels.cu goes through nvcc.
#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "core/problem.hpp"

namespace heat::gpu {

// Kernels that advance the grid by one time step.
enum class StepKernel {
    Naive,   // one thread per point; every value is loaded from global memory (through the caches)
    Shared,  // each block first copies its tile plus a one-point halo into shared memory
};

// Largest number of time steps the temporal kernel does in one launch.
inline constexpr int kMaxTemporalSteps = 8;

// Queues one time step on rows [rows.begin, rows.end), columns 1..N-2, of a row-major array with N columns:
//   uNew = u + alpha * (u_up + u_down + u_left + u_right - 4 u)
// Rows rows.begin - 1 and rows.end must exist (boundary or ghost rows); they are read, not written.
template <typename T>
void launch_step(const T* u, T* uNew, int N, RowRange rows, T alpha, StepKernel kernel = StepKernel::Naive,
                 cudaStream_t stream = nullptr);

// Queues `steps` time steps (1 <= steps <= kMaxTemporalSteps) on the whole N x N grid in a single launch
// (temporal blocking): each block keeps its tile in shared memory for all the steps, so the grid travels
// to and from GPU memory once per launch instead of once per step.
template <typename T>
void launch_temporal(const T* u, T* uNew, int N, int steps, T alpha, cudaStream_t stream = nullptr);

// Sum of n values on the GPU, accumulated in double. Waits for the result.
template <typename T>
double sum(const T* x, std::size_t n, cudaStream_t stream = nullptr);

}  // namespace heat::gpu
