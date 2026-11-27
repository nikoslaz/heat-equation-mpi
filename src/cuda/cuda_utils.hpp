// CUDA helpers: error checking, and RAII wrappers that release GPU resources when they go out of scope.
#pragma once

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

// Exits with CUDA's error message, the failed call and its location if a CUDA runtime call fails.
#define CHECK_CUDA(call)                                                                       \
    do {                                                                                       \
        const cudaError_t heat_cuda_status_ = (call);                                          \
        if (heat_cuda_status_ != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", __FILE__, __LINE__, #call, \
                         cudaGetErrorString(heat_cuda_status_));                               \
            std::exit(EXIT_FAILURE);                                                           \
        }                                                                                      \
    } while (0)

namespace heat::cuda {

// An array in GPU memory.
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) : count_(count) { CHECK_CUDA(cudaMalloc(&data_, bytes())); }
    ~DeviceBuffer() {
        if (data_ != nullptr) cudaFree(data_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), count_(std::exchange(other.count_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        swap(*this, other);
        return *this;
    }

    friend void swap(DeviceBuffer& a, DeviceBuffer& b) noexcept {
        std::swap(a.data_, b.data_);
        std::swap(a.count_, b.count_);
    }

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return count_ * sizeof(T); }

    // Copies a host array of the same size into the buffer.
    void upload(const std::vector<T>& host) {
        if (host.size() != count_) {
            std::fprintf(stderr, "DeviceBuffer::upload: %zu values for a buffer of %zu\n", host.size(), count_);
            std::exit(EXIT_FAILURE);
        }
        CHECK_CUDA(cudaMemcpy(data_, host.data(), bytes(), cudaMemcpyHostToDevice));
    }

    // Copies `count` values starting at `offset` (by default the whole buffer) to the host.
    [[nodiscard]] std::vector<T> download(std::size_t offset = 0, std::size_t count = 0) const {
        if (count == 0) count = count_ - offset;
        std::vector<T> host(count);
        CHECK_CUDA(cudaMemcpy(host.data(), data_ + offset, count * sizeof(T), cudaMemcpyDeviceToHost));
        return host;
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

// Page-locked ("pinned") host memory: the GPU can copy to and from it asynchronously.
template <typename T>
class PinnedBuffer {
public:
    explicit PinnedBuffer(std::size_t count) : count_(count) {
        CHECK_CUDA(cudaMallocHost(&data_, count * sizeof(T)));
    }
    ~PinnedBuffer() {
        if (data_ != nullptr) cudaFreeHost(data_);
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    [[nodiscard]] T* data() noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

// A CUDA stream that does not synchronise with the default stream, so work on different streams can overlap.
class Stream {
public:
    Stream() { CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
    void synchronize() const { CHECK_CUDA(cudaStreamSynchronize(stream_)); }

private:
    cudaStream_t stream_ = nullptr;
};

// A CUDA event, for timing work on the GPU.
class Event {
public:
    Event() { CHECK_CUDA(cudaEventCreate(&event_)); }
    ~Event() { cudaEventDestroy(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;

    void record(cudaStream_t stream = nullptr) { CHECK_CUDA(cudaEventRecord(event_, stream)); }
    void synchronize() const { CHECK_CUDA(cudaEventSynchronize(event_)); }

    // Seconds between two events that have both completed.
    static double seconds_between(const Event& start, const Event& stop) {
        float ms = 0.0F;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start.event_, stop.event_));
        return ms / 1000.0;
    }

private:
    cudaEvent_t event_ = nullptr;
};

struct DeviceInfo {
    std::string name;
    double peakBandwidthGBs = 0.0;  // theoretical peak memory bandwidth, 0 if the driver does not report it
};

inline DeviceInfo current_device() {
    int device = 0;
    CHECK_CUDA(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    DeviceInfo info{prop.name, 0.0};

    int clockKHz = 0;
    int busBits = 0;
    if (cudaDeviceGetAttribute(&clockKHz, cudaDevAttrMemoryClockRate, device) == cudaSuccess &&
        cudaDeviceGetAttribute(&busBits, cudaDevAttrGlobalMemoryBusWidth, device) == cudaSuccess) {
        info.peakBandwidthGBs = 2.0 * clockKHz * 1e3 * (busBits / 8.0) / 1e9;  // two transfers per clock (DDR)
    } else {
        (void)cudaGetLastError();  // clear the error
    }
    return info;
}

}  // namespace heat::cuda
