// CUDA kernels for the 2D heat equation. See heat_gpu.hpp for the host-side interface.
#include "heat_gpu.hpp"
#include "cuda_check.hpp"

#include <algorithm>
#include <vector>

namespace {

// ---------------------------------------------------------------------------------------------
// One time step per launch
// ---------------------------------------------------------------------------------------------

constexpr int kBX = 32;   // threads per block along a row (32 = one warp, so loads are coalesced)
constexpr int kBY = 8;    // threads per block across rows

// One thread per grid point. Neighbouring threads load overlapping values; the L1/L2 caches
// serve most of the repeated loads, so each value comes from DRAM roughly once per step.
template <typename T>
__global__ void step_naive(const T* __restrict__ u, T* __restrict__ uNew,
                           int N, int rowBegin, int rowEnd, T alpha) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = rowBegin + blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rowEnd || j < 1 || j >= N - 1) return;

    const size_t idx = (size_t)i * N + j;
    const T c = u[idx];
    uNew[idx] = c + alpha * (u[idx - N] + u[idx + N] + u[idx - 1] + u[idx + 1] - T(4) * c);
}

// The block copies its kBY x kBX tile plus a one-point halo into shared memory, then every
// thread reads its four neighbours from there. Rows rowBegin-1 .. rowEnd exist in memory.
template <typename T>
__global__ void step_shared(const T* __restrict__ u, T* __restrict__ uNew,
                            int N, int rowBegin, int rowEnd, T alpha) {
    __shared__ T s[kBY + 2][kBX + 2];

    const int tx = threadIdx.x, ty = threadIdx.y;
    const int j = blockIdx.x * kBX + tx;
    const int i = rowBegin + blockIdx.y * kBY + ty;

    if (j < N) {
        if (i <= rowEnd) s[ty + 1][tx + 1] = u[(size_t)i * N + j];
        if (ty == 0) s[0][tx + 1] = u[(size_t)(i - 1) * N + j];
        if (ty == kBY - 1 && i + 1 <= rowEnd) s[kBY + 1][tx + 1] = u[(size_t)(i + 1) * N + j];
    }
    if (i <= rowEnd) {
        if (tx == 0 && j >= 1) s[ty + 1][0] = u[(size_t)i * N + j - 1];
        if (tx == kBX - 1 && j + 1 < N) s[ty + 1][kBX + 1] = u[(size_t)i * N + j + 1];
    }
    __syncthreads();

    if (i < rowEnd && j >= 1 && j < N - 1) {
        const T c = s[ty + 1][tx + 1];
        uNew[(size_t)i * N + j] =
            c + alpha * (s[ty][tx + 1] + s[ty + 2][tx + 1] + s[ty + 1][tx] + s[ty + 1][tx + 2] - T(4) * c);
    }
}

// ---------------------------------------------------------------------------------------------
// Several time steps per launch (temporal blocking)
// ---------------------------------------------------------------------------------------------

constexpr int kTX = 64;                    // tile width (columns)
constexpr int kTY = 32;                    // tile height (rows)
constexpr int kTRows = 4;                  // rows handled by each thread
constexpr int kTThreadsY = kTY / kTRows;   // block = kTX x kTThreadsY = 512 threads

// Each block loads a kTY x kTX tile into shared memory and advances it S steps there.
// After s steps only the points at least s away from the tile edge are still exact (the edge
// has no neighbours outside the tile), so the block writes back the inner
// (kTY - 2S) x (kTX - 2S) points. Neighbouring tiles overlap by 2S rows/columns, and the
// points in the overlap are computed by both blocks: extra arithmetic, but S times less
// DRAM traffic, which is the bottleneck for this stencil.
template <typename T>
__global__ void __launch_bounds__(kTX * kTThreadsY)
step_temporal(const T* __restrict__ u, T* __restrict__ uNew, int N, int S, T alpha) {
    __shared__ T buf[2][kTY][kTX];

    const int outX = kTX - 2 * S;              // points written per block along a row
    const int outY = kTY - 2 * S;
    const int j0 = blockIdx.x * outX - S;      // global column of tile column 0
    const int i0 = blockIdx.y * outY - S;      // global row of tile row 0
    const int c = threadIdx.x;
    const int gj = j0 + c;

#pragma unroll
    for (int k = 0; k < kTRows; k++) {
        const int r = threadIdx.y + k * kTThreadsY;
        const int gi = i0 + r;
        buf[0][r][c] = (gi >= 0 && gi < N && gj >= 0 && gj < N) ? u[(size_t)gi * N + gj] : T(0);
    }
    __syncthreads();

    for (int s = 0; s < S; s++) {
        const T (*src)[kTX] = buf[s & 1];
        T (*dst)[kTX] = buf[(s + 1) & 1];
#pragma unroll
        for (int k = 0; k < kTRows; k++) {
            const int r = threadIdx.y + k * kTThreadsY;
            const int gi = i0 + r;
            T v = src[r][c];
            // Only interior points of the grid change; the fixed boundary and the points
            // outside the grid keep their values.
            if (r >= 1 && r < kTY - 1 && c >= 1 && c < kTX - 1 &&
                gi >= 1 && gi < N - 1 && gj >= 1 && gj < N - 1) {
                v = v + alpha * (src[r - 1][c] + src[r + 1][c] + src[r][c - 1] + src[r][c + 1] - T(4) * v);
            }
            dst[r][c] = v;
        }
        __syncthreads();
    }

    const T (*fin)[kTX] = buf[S & 1];
    if (c >= S && c < kTX - S && gj < N) {
#pragma unroll
        for (int k = 0; k < kTRows; k++) {
            const int r = threadIdx.y + k * kTThreadsY;
            const int gi = i0 + r;
            if (r >= S && r < kTY - S && gi < N) uNew[(size_t)gi * N + gj] = fin[r][c];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Reduction
// ---------------------------------------------------------------------------------------------

constexpr int kSumThreads = 256;
constexpr int kSumMaxBlocks = 1024;

// Each block sums a strided part of x: per-thread partial sums, then a reduction inside each
// warp with shuffles, then across the block's warps. One partial sum per block.
template <typename T>
__global__ void sum_partial(const T* __restrict__ x, size_t n, double* __restrict__ partial) {
    double v = 0.0;
    for (size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x; k < n; k += (size_t)gridDim.x * blockDim.x)
        v += (double)x[k];

    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);

    __shared__ double warpSums[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) warpSums[warp] = v;
    __syncthreads();

    if (warp == 0) {
        v = (lane < (int)(blockDim.x >> 5)) ? warpSums[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffffu, v, offset);
        if (lane == 0) partial[blockIdx.x] = v;
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------------------------

template <typename T>
void launch_step_rows(const T* u, T* uNew, int N, int rowBegin, int rowEnd, T alpha,
                      StepKernel kernel, cudaStream_t stream) {
    if (rowEnd <= rowBegin) return;
    const dim3 block(kBX, kBY);
    const dim3 grid((N + kBX - 1) / kBX, (rowEnd - rowBegin + kBY - 1) / kBY);
    if (kernel == StepKernel::Shared)
        step_shared<T><<<grid, block, 0, stream>>>(u, uNew, N, rowBegin, rowEnd, alpha);
    else
        step_naive<T><<<grid, block, 0, stream>>>(u, uNew, N, rowBegin, rowEnd, alpha);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch_temporal(const T* u, T* uNew, int N, int S, T alpha, cudaStream_t stream) {
    if (S < 1 || S > kMaxTemporalSteps) {
        std::fprintf(stderr, "launch_temporal: S must be between 1 and %d\n", kMaxTemporalSteps);
        std::exit(EXIT_FAILURE);
    }
    const int outX = kTX - 2 * S, outY = kTY - 2 * S;
    const dim3 block(kTX, kTThreadsY);
    const dim3 grid((N + outX - 1) / outX, (N + outY - 1) / outY);
    step_temporal<T><<<grid, block, 0, stream>>>(u, uNew, N, S, alpha);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
double gpu_sum(const T* x, size_t n, cudaStream_t stream) {
    if (n == 0) return 0.0;
    const int blocks = (int)std::min<size_t>(kSumMaxBlocks, (n + kSumThreads - 1) / kSumThreads);

    double* dPartial = nullptr;
    CUDA_CHECK(cudaMalloc(&dPartial, blocks * sizeof(double)));
    sum_partial<T><<<blocks, kSumThreads, 0, stream>>>(x, n, dPartial);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> partial(blocks);
    CUDA_CHECK(cudaMemcpyAsync(partial.data(), dPartial, blocks * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFree(dPartial));

    double total = 0.0;
    for (double p : partial) total += p;   // fixed order, so the result is reproducible
    return total;
}

template void launch_step_rows<float>(const float*, float*, int, int, int, float, StepKernel, cudaStream_t);
template void launch_step_rows<double>(const double*, double*, int, int, int, double, StepKernel, cudaStream_t);
template void launch_temporal<float>(const float*, float*, int, int, float, cudaStream_t);
template void launch_temporal<double>(const double*, double*, int, int, double, cudaStream_t);
template double gpu_sum<float>(const float*, size_t, cudaStream_t);
template double gpu_sum<double>(const double*, size_t, cudaStream_t);
