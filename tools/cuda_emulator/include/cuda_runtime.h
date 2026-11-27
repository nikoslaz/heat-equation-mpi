// A CPU emulation of the small part of CUDA this project uses, so the CUDA code can be built and tested on
// machines without an NVIDIA GPU (a Mac, a CI runner). See README.md in tools/cuda_emulator.
//
// Kernels run on a pool of CPU threads, one per CUDA thread of a block. Blocks run one at a time and in a
// shuffled order, because a GPU guarantees no order either: a kernel whose result depends on the order of its
// blocks (a race between blocks) gives different results here. Streams and memory copies are synchronous.
#pragma once

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

// ---- Language extensions --------------------------------------------------------------------------------

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
// Only one block runs at a time, so a block's shared memory can be a static variable.
#define __shared__ static

struct dim3 {
    unsigned x, y, z;
    constexpr dim3(unsigned x0 = 1, unsigned y0 = 1, unsigned z0 = 1) : x(x0), y(y0), z(z0) {}
};

inline thread_local dim3 threadIdx;
inline thread_local dim3 blockIdx;
inline thread_local dim3 blockDim;
inline thread_local dim3 gridDim;

// ---- Runtime API ----------------------------------------------------------------------------------------

using cudaError_t = int;
inline constexpr cudaError_t cudaSuccess = 0;
inline constexpr cudaError_t cudaErrorNotSupported = 801;

inline const char* cudaGetErrorString(cudaError_t status) {
    return status == cudaSuccess ? "no error" : "not supported by the CPU emulator";
}
inline cudaError_t cudaGetLastError() {
    return cudaSuccess;
}

struct EmuStream;
using cudaStream_t = EmuStream*;
inline constexpr unsigned cudaStreamNonBlocking = 1;

enum cudaMemcpyKind {
    cudaMemcpyHostToHost,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
    cudaMemcpyDefault
};

struct cudaDeviceProp {
    char name[256];
};

enum cudaDeviceAttr { cudaDevAttrMemoryClockRate = 36, cudaDevAttrGlobalMemoryBusWidth = 37 };

inline cudaError_t cudaGetDeviceCount(int* count) {
    *count = 1;
    return cudaSuccess;
}
inline cudaError_t cudaSetDevice(int) {
    return cudaSuccess;
}
inline cudaError_t cudaGetDevice(int* device) {
    *device = 0;
    return cudaSuccess;
}
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int) {
    std::snprintf(prop->name, sizeof(prop->name), "CPU emulator");
    return cudaSuccess;
}
// There is no meaningful memory bandwidth to report.
inline cudaError_t cudaDeviceGetAttribute(int* value, cudaDeviceAttr, int) {
    *value = 0;
    return cudaErrorNotSupported;
}
inline cudaError_t cudaDeviceSynchronize() {
    return cudaSuccess;
}

// New allocations are filled with 0xFF bytes (NaN as double or float), so a kernel that reads memory nobody
// wrote produces NaN, which --verify reports.
template <typename T>
cudaError_t cudaMalloc(T** pointer, std::size_t bytes) {
    void* memory = std::malloc(bytes > 0 ? bytes : 1);
    std::memset(memory, 0xFF, bytes);
    *pointer = static_cast<T*>(memory);
    return cudaSuccess;
}
template <typename T>
cudaError_t cudaMallocHost(T** pointer, std::size_t bytes) {
    return cudaMalloc(pointer, bytes);
}
inline cudaError_t cudaFree(void* pointer) {
    std::free(pointer);
    return cudaSuccess;
}
inline cudaError_t cudaFreeHost(void* pointer) {
    return cudaFree(pointer);
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind) {
    std::memcpy(dst, src, bytes);
    return cudaSuccess;
}
inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t bytes, cudaMemcpyKind kind,
                                   cudaStream_t = nullptr) {
    return cudaMemcpy(dst, src, bytes, kind);
}

inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* stream, unsigned) {
    *stream = nullptr;
    return cudaSuccess;
}
inline cudaError_t cudaStreamDestroy(cudaStream_t) {
    return cudaSuccess;
}
inline cudaError_t cudaStreamSynchronize(cudaStream_t) {
    return cudaSuccess;
}

struct EmuEvent {
    std::chrono::steady_clock::time_point time;
};
using cudaEvent_t = EmuEvent*;

inline cudaError_t cudaEventCreate(cudaEvent_t* event) {
    *event = new EmuEvent{};
    return cudaSuccess;
}
inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
    delete event;
    return cudaSuccess;
}
inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t = nullptr) {
    event->time = std::chrono::steady_clock::now();
    return cudaSuccess;
}
inline cudaError_t cudaEventSynchronize(cudaEvent_t) {
    return cudaSuccess;
}
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t stop) {
    *ms = std::chrono::duration<float, std::milli>(stop->time - start->time).count();
    return cudaSuccess;
}

// ---- Kernel execution -----------------------------------------------------------------------------------

namespace emu {

inline thread_local unsigned threadRank = 0;  // the CUDA thread's index within its block

// A barrier for many more threads than cores. Waiting threads yield the CPU instead of sleeping, which is much
// faster here than std::barrier: every phase would otherwise put hundreds of threads to sleep and wake them up.
class Barrier {
public:
    explicit Barrier(unsigned count) : count_(count) {}

    void arrive_and_wait() {
        const unsigned generation = generation_.load(std::memory_order_acquire);
        if (arrived_.fetch_add(1, std::memory_order_acq_rel) + 1 == count_) {
            arrived_.store(0, std::memory_order_relaxed);
            generation_.fetch_add(1, std::memory_order_release);
            return;
        }
        while (generation_.load(std::memory_order_acquire) == generation) std::this_thread::yield();
    }

private:
    const unsigned count_;
    std::atomic<unsigned> arrived_{0};
    std::atomic<unsigned> generation_{0};
};

// Runs kernel launches block by block.
//
// The first launch of a kernel runs on a pool with one std::thread per CUDA thread of a block, so that
// __syncthreads and warp shuffles work. If the kernel used neither, later launches of it run every CUDA thread
// one after the other on the calling thread, which is much faster and gives the same result: threads that
// never wait for each other can run in any order.
class Executor {
public:
    using Key = void (*)();  // identifies a kernel

    static Executor& get() {
        static Executor executor;
        return executor;
    }
    Executor(const Executor&) = delete;
    Executor& operator=(const Executor&) = delete;
    ~Executor() { resize(0); }

    void run(Key kernel, dim3 grid, dim3 block, std::function<void()> body) {
        const auto known = synchronises_.find(kernel);
        if (known != synchronises_.end() && !known->second) {
            run_sequentially(grid, block, body);
            return;
        }
        synchronised_ = false;
        run_on_threads(grid, block, std::move(body));
        synchronises_[kernel] = synchronised_;
    }

    void sync_threads() {
        check_not_sequential("__syncthreads");
        synchronised_ = true;
        sync_->arrive_and_wait();
    }

    template <typename T>
    T shuffle_down(T value, unsigned delta) {
        static_assert(sizeof(T) <= sizeof(std::uint64_t), "values up to 8 bytes");
        check_not_sequential("__shfl_down_sync");
        synchronised_ = true;
        const unsigned t = threadRank;
        const unsigned lane = t % 32;
        Barrier& warp = *warps_[t / 32];
        std::memcpy(&lanes_[t], &value, sizeof(T));
        warp.arrive_and_wait();
        T result = value;  // lanes past the end of the warp keep their own value, as on a GPU
        if (lane + delta < 32 && t + delta < lanes_.size()) std::memcpy(&result, &lanes_[t + delta], sizeof(T));
        warp.arrive_and_wait();
        return result;
    }

private:
    Executor() = default;

    void run_on_threads(dim3 grid, dim3 block, std::function<void()> body) {
        const unsigned threads = block.x * block.y * block.z;
        if (threads != workers_.size()) resize(threads);
        grid_ = grid;
        block_ = block;
        body_ = std::move(body);
        order_ = block_order(grid);
        start_->arrive_and_wait();   // the workers run every block ...
        finish_->arrive_and_wait();  // ... and are done
    }

    void run_sequentially(dim3 grid, dim3 block, const std::function<void()>& body) {
        sequential_ = true;
        gridDim = grid;
        blockDim = block;
        for (const dim3& b : block_order(grid)) {
            blockIdx = b;
            unsigned rank = 0;
            for (unsigned z = 0; z < block.z; z++) {
                for (unsigned y = 0; y < block.y; y++) {
                    for (unsigned x = 0; x < block.x; x++) {
                        threadIdx = dim3(x, y, z);
                        threadRank = rank++;
                        body();
                    }
                }
            }
        }
        sequential_ = false;
    }

    void check_not_sequential(const char* what) const {
        if (sequential_) {
            std::fprintf(stderr,
                         "CUDA emulator: a kernel called %s, but its first launch did not; the emulator cannot "
                         "run kernels whose use of %s depends on their arguments\n",
                         what, what);
            std::abort();
        }
    }

    void resize(unsigned threads) {
        if (!workers_.empty()) {
            stop_ = true;
            start_->arrive_and_wait();
            for (std::thread& w : workers_) w.join();
            workers_.clear();
            stop_ = false;
        }
        if (threads == 0) return;
        start_ = std::make_unique<Barrier>(threads + 1);
        finish_ = std::make_unique<Barrier>(threads + 1);
        sync_ = std::make_unique<Barrier>(threads);
        warps_.clear();
        for (unsigned w = 0; w * 32 < threads; w++) {
            warps_.push_back(std::make_unique<Barrier>(std::min(32U, threads - 32 * w)));
        }
        lanes_.assign(warps_.size() * 32, 0);
        for (unsigned id = 0; id < threads; id++) workers_.emplace_back([this, id] { work(id); });
    }

    void work(unsigned id) {
        threadRank = id;
        for (;;) {
            start_->arrive_and_wait();
            if (stop_) return;
            gridDim = grid_;
            blockDim = block_;
            threadIdx = dim3(id % block_.x, (id / block_.x) % block_.y, id / (block_.x * block_.y));
            for (const dim3& b : order_) {
                blockIdx = b;
                body_();
                sync_->arrive_and_wait();  // the whole block is done before its shared memory is reused
            }
            finish_->arrive_and_wait();
        }
    }

    // CUDA_EMU_ORDER=forward|reverse|shuffle (default shuffle, with a fixed seed so runs are reproducible).
    std::vector<dim3> block_order(dim3 grid) {
        std::vector<dim3> order;
        for (unsigned z = 0; z < grid.z; z++) {
            for (unsigned y = 0; y < grid.y; y++) {
                for (unsigned x = 0; x < grid.x; x++) order.emplace_back(x, y, z);
            }
        }
        static const std::string mode = std::getenv("CUDA_EMU_ORDER") ? std::getenv("CUDA_EMU_ORDER") : "shuffle";
        if (mode == "reverse") std::reverse(order.begin(), order.end());
        if (mode == "shuffle") std::shuffle(order.begin(), order.end(), rng_);
        return order;
    }

    std::vector<std::thread> workers_;
    std::unique_ptr<Barrier> start_;
    std::unique_ptr<Barrier> finish_;
    std::unique_ptr<Barrier> sync_;
    std::vector<std::unique_ptr<Barrier>> warps_;
    std::vector<std::uint64_t> lanes_;
    dim3 grid_;
    dim3 block_;
    std::vector<dim3> order_;
    std::function<void()> body_;
    bool stop_ = false;
    bool sequential_ = false;
    std::atomic<bool> synchronised_{false};
    std::map<Key, bool> synchronises_;  // whether a kernel used __syncthreads or a shuffle in its first launch
    std::mt19937 rng_{12345};
};

struct LaunchConfig {
    dim3 grid;
    dim3 block;
};

inline LaunchConfig cfg(dim3 grid, dim3 block, std::size_t = 0, cudaStream_t = nullptr) {
    return {grid, block};
}

// What "kernel<<<config>>>(args...)" becomes (see convert_launches.cmake).
template <typename... Params, typename... Args>
void launch(LaunchConfig config, void (*kernel)(Params...), Args&&... args) {
    // Copy the arguments, as a real kernel launch does.
    std::tuple<std::decay_t<Params>...> copied(std::forward<Args>(args)...);
    Executor::get().run(reinterpret_cast<Executor::Key>(kernel), config.grid, config.block,
                        [kernel, copied] { std::apply(kernel, copied); });
}

}  // namespace emu

inline void __syncthreads() {
    emu::Executor::get().sync_threads();
}

template <typename T>
T __shfl_down_sync(unsigned /*mask*/, T value, unsigned delta) {
    return emu::Executor::get().shuffle_down(value, delta);
}
