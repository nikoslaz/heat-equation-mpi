#include "heat_cuda.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                    \
                         cudaGetErrorString(err_), __FILE__, __LINE__);         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// One thread per interior point. Boundary rows/columns are never written,
// so they keep their initial values in both buffers.
__global__ void heat_step(const double* __restrict__ u, double* __restrict__ uNew,
                          int N, double alpha) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;   // column (contiguous in memory)
    int i = blockIdx.y * blockDim.y + threadIdx.y;   // row
    if (i < 1 || i >= N - 1 || j < 1 || j >= N - 1) return;

    int idx = i * N + j;
    uNew[idx] = u[idx] + alpha * (u[idx - N] + u[idx + N] + u[idx - 1] + u[idx + 1] - 4.0 * u[idx]);
}

void run_cuda(int K, int N) {
    if (N < 3) {
        std::cerr << "Error: N must be at least 3\n";
        return;
    }

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << std::endl;

    constexpr double alpha = 0.2;
    const size_t count = (size_t)N * N;
    const size_t bytes = count * sizeof(double);

    // Initial condition: top row = 100, everything else 0.
    std::vector<double> h(count, 0.0);
    for (int j = 0; j < N; j++) h[j] = 100.0;

    double *dU = nullptr, *dUNew = nullptr;
    CUDA_CHECK(cudaMalloc(&dU, bytes));
    CUDA_CHECK(cudaMalloc(&dUNew, bytes));
    CUDA_CHECK(cudaMemcpy(dU,    h.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dUNew, h.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(32, 8);
    const dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int step = 0; step < K; step++) {
        heat_step<<<grid, block>>>(dU, dUNew, N, alpha);
        std::swap(dU, dUNew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h.data(), dU, bytes, cudaMemcpyDeviceToHost));

    double total = 0.0;
    for (size_t k = 0; k < count; k++) total += h[k];

    if (N <= 1024) {
        std::ofstream f("grid.csv");
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                f << h[(size_t)i * N + j] << (j + 1 < N ? "," : "\n");
    }

    std::cout << "Average = " << total / ((double)N * N) << std::endl;
    std::cout << "Time = " << ms / 1000.0 << " s" << std::endl;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dU));
    CUDA_CHECK(cudaFree(dUNew));
}
