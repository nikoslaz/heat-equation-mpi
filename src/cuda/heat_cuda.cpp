// heat_cuda: the 2D heat equation on one GPU. The kernels are in kernels.cu.
#include <algorithm>
#include <cmath>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "core/cli.hpp"
#include "core/report.hpp"
#include "cuda/cuda_utils.hpp"
#include "cuda/kernels.hpp"

namespace {

using heat::cuda::DeviceBuffer;

enum class Kernel { Naive, Shared, Temporal };

struct Settings {
    Kernel kernel = Kernel::Naive;
    int tsteps = 4;  // time steps per launch of the temporal kernel
    std::string description;
};

Kernel kernel_from_name(const std::string& name) {
    if (name == "shared") return Kernel::Shared;
    if (name == "temporal") return Kernel::Temporal;
    return Kernel::Naive;
}

// Advances the grid by `steps` time steps; u holds the newest grid afterwards.
template <typename T>
void advance(DeviceBuffer<T>& u, DeviceBuffer<T>& uNew, int N, int steps, const Settings& s) {
    const T alpha = static_cast<T>(heat::kAlpha);
    if (s.kernel == Kernel::Temporal) {
        for (int done = 0; done < steps;) {
            const int n = std::min(s.tsteps, steps - done);  // the last launch may do fewer steps
            heat::gpu::launch_temporal(u.data(), uNew.data(), N, n, alpha);
            swap(u, uNew);
            done += n;
        }
        return;
    }
    const auto kernel = s.kernel == Kernel::Shared ? heat::gpu::StepKernel::Shared : heat::gpu::StepKernel::Naive;
    for (int step = 0; step < steps; step++) {
        heat::gpu::launch_step(u.data(), uNew.data(), N, {1, N - 1}, alpha, kernel);
        swap(u, uNew);
    }
}

void print_bandwidth(std::ostream& out, const heat::Problem& p, double seconds, std::size_t bytesPerValue,
                     double peakGBs) {
    if (seconds <= 0) return;
    // The memory traffic a kernel that reads and writes the grid once per step would need for this rate.
    const double effective = heat::glups(p, seconds) * 2.0 * static_cast<double>(bytesPerValue);
    std::string text = heat::format_number(effective) + " GB/s effective";
    if (peakGBs > 0) {
        text += " (" + std::to_string(std::lround(100.0 * effective / peakGBs)) + "% of the " +
                heat::format_number(peakGBs) + " GB/s peak)";
    }
    heat::print_field(out, "Bandwidth", text);
}

template <typename T>
int run(const heat::Problem& p, const Settings& s, const heat::OutputOptions& output) {
    const int N = p.N;
    const std::vector<T> initial = heat::initial_grid<T>(N);
    DeviceBuffer<T> u(initial.size());
    DeviceBuffer<T> uNew(initial.size());

    // Untimed warm-up step: the first launch of a kernel also loads it onto the GPU.
    u.upload(initial);
    uNew.upload(initial);
    advance(u, uNew, N, 1, s);
    CHECK_CUDA(cudaDeviceSynchronize());
    u.upload(initial);
    uNew.upload(initial);

    heat::cuda::Event start;
    heat::cuda::Event stop;
    start.record();
    advance(u, uNew, N, p.K, s);
    stop.record();
    stop.synchronize();
    const double seconds = heat::cuda::Event::seconds_between(start, stop);
    const double average = heat::gpu::sum(u.data(), u.size()) / (static_cast<double>(N) * N);

    const heat::cuda::DeviceInfo device = heat::cuda::current_device();
    std::ostream& out = std::cout;
    heat::print_field(out, "Program", "heat_cuda (one GPU)");
    heat::print_field(out, "Grid", heat::describe(p));
    heat::print_field(out, "Setup", s.description);
    heat::print_field(out, "Device", device.name);
    heat::print_timing(out, p, average, seconds);
    print_bandwidth(out, p, seconds, sizeof(T), device.peakBandwidthGBs);

    int status = 0;
    const std::string csv = output.csv_path(N);
    if (output.verify || !csv.empty()) {
        const std::vector<T> grid = u.download();
        if (output.verify && !heat::verify_against_serial(out, p, grid, average)) status = 1;
        if (!csv.empty() && !heat::write_csv(out, csv, grid, N)) status = 1;
    }
    return status;
}

}  // namespace

int main(int argc, char** argv) {
    heat::Problem problem;
    heat::OutputOptions output;
    Settings settings;
    std::string kernel = "naive";
    std::string precision = "double";

    heat::ArgParser args("heat_cuda", "2D heat equation on one GPU (CUDA).");
    args.add_choice("--kernel", {"naive", "shared", "temporal"}, "how the GPU computes the time steps", &kernel);
    args.add_choice("--precision", {"double", "float"}, "floating-point type of the grid", &precision);
    args.add_int("--tsteps", "S", "time steps per launch of the temporal kernel", &settings.tsteps, 1,
                 heat::gpu::kMaxTemporalSteps);
    heat::add_output_options(args, output);
    switch (args.parse(argc, argv, problem, std::cout, std::cerr)) {
        case heat::ArgParser::Result::Help: return 0;
        case heat::ArgParser::Result::Error: return 1;
        case heat::ArgParser::Result::Ok: break;
    }

    settings.kernel = kernel_from_name(kernel);
    settings.description = kernel + " kernel";
    if (settings.kernel == Kernel::Temporal)
        settings.description += " (S = " + std::to_string(settings.tsteps) + ")";
    settings.description += ", " + precision + " precision";

    return precision == "float" ? run<float>(problem, settings, output) : run<double>(problem, settings, output);
}
