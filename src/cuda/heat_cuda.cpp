// Single-GPU 2D heat solver: argument parsing, timing and output. The kernels are in heat_gpu.cu.
#include "cuda_check.hpp"
#include "heat_gpu.hpp"
#include "heat_reference.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
    int K = 0;
    int N = 0;
    std::string kernel = "naive";      // naive | shared | temporal
    std::string precision = "double";  // double | float
    int tsteps = 4;                    // time steps per launch for the temporal kernel
    bool verify = false;               // compare every grid point with a CPU run
    bool csv = true;                   // write grid.csv when N <= 1024
};

void usage() {
    std::cout << "Usage: heat_cuda K N [options]\n"
                 "  K                 number of time steps\n"
                 "  N                 grid size (N x N, N >= 3)\n"
                 "  --kernel NAME     naive (default), shared or temporal\n"
                 "  --precision P     double (default) or float\n"
                 "  --tsteps S        time steps per launch for --kernel temporal (1-"
              << kMaxTemporalSteps << ", default 4)\n"
                 "  --verify          compare every grid point with a CPU run of the same problem\n"
                 "  --no-csv          do not write grid.csv\n";
}

bool parse_args(int argc, char** argv, Options& o) {
    if (argc < 3) return false;
    try {
        o.K = std::stoi(argv[1]);
        o.N = std::stoi(argv[2]);
        for (int a = 3; a < argc; a++) {
            const std::string arg = argv[a];
            const bool hasValue = a + 1 < argc;
            if (arg == "--kernel" && hasValue) o.kernel = argv[++a];
            else if (arg == "--precision" && hasValue) o.precision = argv[++a];
            else if (arg == "--tsteps" && hasValue) o.tsteps = std::stoi(argv[++a]);
            else if (arg == "--verify") o.verify = true;
            else if (arg == "--no-csv") o.csv = false;
            else return false;
        }
    } catch (...) {
        return false;
    }
    const bool kernelOk = o.kernel == "naive" || o.kernel == "shared" || o.kernel == "temporal";
    const bool precisionOk = o.precision == "double" || o.precision == "float";
    return o.K >= 0 && o.N >= 3 && kernelOk && precisionOk && o.tsteps >= 1 && o.tsteps <= kMaxTemporalSteps;
}

// Theoretical peak memory bandwidth in GB/s, or 0 if the driver does not report it.
double peak_bandwidth_gbs(int device) {
    int clockKHz = 0, busBits = 0;
    if (cudaDeviceGetAttribute(&clockKHz, cudaDevAttrMemoryClockRate, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&busBits, cudaDevAttrGlobalMemoryBusWidth, device) != cudaSuccess) {
        cudaGetLastError();   // clear the error
        return 0.0;
    }
    return 2.0 * clockKHz * 1e3 * (busBits / 8.0) / 1e9;   // two transfers per clock (DDR)
}

// Advance the grid by `steps` time steps. dU always points to the newest grid afterwards.
template <typename T>
void advance(T*& dU, T*& dUNew, int N, int steps, T alpha, const Options& o) {
    if (o.kernel == "temporal") {
        for (int done = 0; done < steps;) {
            const int s = std::min(o.tsteps, steps - done);   // the last launch may do fewer steps
            launch_temporal(dU, dUNew, N, s, alpha);
            std::swap(dU, dUNew);
            done += s;
        }
    } else {
        const StepKernel kernel = (o.kernel == "shared") ? StepKernel::Shared : StepKernel::Naive;
        for (int step = 0; step < steps; step++) {
            launch_step_rows(dU, dUNew, N, 1, N - 1, alpha, kernel);
            std::swap(dU, dUNew);
        }
    }
}

template <typename T>
int run(const Options& o, double peakGBs) {
    const int K = o.K, N = o.N;
    const size_t count = (size_t)N * N;
    const size_t bytes = count * sizeof(T);
    const T alpha = T(0.2);

    // Initial condition: top row = 100, everything else 0.
    std::vector<T> h(count, T(0));
    for (int j = 0; j < N; j++) h[j] = T(100);

    T *dU = nullptr, *dUNew = nullptr;
    CUDA_CHECK(cudaMalloc(&dU, bytes));
    CUDA_CHECK(cudaMalloc(&dUNew, bytes));

    // Untimed warm-up step: the first launch of a kernel also loads it onto the GPU.
    CUDA_CHECK(cudaMemcpy(dU, h.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dUNew, h.data(), bytes, cudaMemcpyHostToDevice));
    advance(dU, dUNew, N, 1, alpha, o);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(dU, h.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dUNew, h.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    advance(dU, dUNew, N, K, alpha, o);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    const double seconds = ms / 1000.0;

    const double average = gpu_sum(dU, count) / ((double)N * N);

    const bool writeCsv = o.csv && N <= 1024;
    if (writeCsv || o.verify) CUDA_CHECK(cudaMemcpy(h.data(), dU, bytes, cudaMemcpyDeviceToHost));

    if (writeCsv) {
        std::ofstream f("grid.csv");
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                f << h[(size_t)i * N + j] << (j + 1 < N ? "," : "\n");
    }

    std::string label = o.kernel;
    if (o.kernel == "temporal") label += " (S=" + std::to_string(o.tsteps) + ")";
    std::cout << "Kernel: " << label << ", precision: " << o.precision << "\n";
    std::cout << "Average = " << average << "\n";
    std::cout << "Time = " << seconds << " s\n";
    if (seconds > 0) {
        // Lattice updates per second, and the DRAM traffic a one-step-per-launch kernel would
        // need for that rate (one read + one write per point per step).
        const double glups = (double)N * N * K / seconds / 1e9;
        std::cout << "Throughput = " << glups << " GLUP/s\n";
        std::cout << "Bandwidth = " << glups * 2 * sizeof(T) << " GB/s effective";
        if (peakGBs > 0) std::cout << " (device peak " << peakGBs << " GB/s)";
        std::cout << "\n";
    }

    int status = 0;
    if (o.verify) {
        const std::vector<double> ref = heat_reference(K, N);
        double maxDiff = 0.0;
        for (size_t k = 0; k < count; k++) {
            const double d = std::fabs((double)h[k] - ref[k]);
            if (!(d <= maxDiff)) maxDiff = d;   // also catches NaN
        }
        const double tol = (sizeof(T) == sizeof(double)) ? 1e-9 : 1e-2;
        const bool ok = maxDiff <= tol;
        std::cout << "Verify: max |gpu - cpu| = " << maxDiff << " (tolerance " << tol << ") "
                  << (ok ? "OK" : "FAILED") << "\n";
        status = ok ? 0 : 1;
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dU));
    CUDA_CHECK(cudaFree(dUNew));
    return status;
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    if (!parse_args(argc, argv, o)) {
        usage();
        return 1;
    }

    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "GPU: " << prop.name << "\n";

    const double peak = peak_bandwidth_gbs(device);
    return (o.precision == "float") ? run<float>(o, peak) : run<double>(o, peak);
}
