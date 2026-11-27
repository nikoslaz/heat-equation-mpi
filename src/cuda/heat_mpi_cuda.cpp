// heat_mpi_cuda: the 2D heat equation on several GPUs. Same row decomposition as heat, but each rank keeps
// its strip and ghost rows in the memory of its own GPU. The kernels are in kernels.cu.
#include <algorithm>
#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include <mpi.h>

#include "core/cli.hpp"
#include "core/report.hpp"
#include "cuda/cuda_utils.hpp"
#include "cuda/kernels.hpp"
#include "mpi/mpi_utils.hpp"

namespace {

using heat::cuda::DeviceBuffer;
using heat::cuda::PinnedBuffer;
using heat::cuda::Stream;

constexpr int kTagUp = 0;    // a row travelling to the rank above
constexpr int kTagDown = 1;  // a row travelling to the rank below

struct Settings {
    bool overlap = true;     // compute the inner rows while the ghost rows are exchanged
    bool cudaAware = false;  // pass GPU memory straight to MPI (needs an MPI built with CUDA support)
};

// Gives each rank on a node its own GPU (round-robin if there are more ranks than GPUs) and returns the
// number of GPUs on the node.
int select_gpu(MPI_Comm comm) {
    MPI_Comm node = MPI_COMM_NULL;
    CHECK_MPI(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node));
    int nodeRank = 0;
    CHECK_MPI(MPI_Comm_rank(node, &nodeRank));
    CHECK_MPI(MPI_Comm_free(&node));

    int count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&count));
    CHECK_CUDA(cudaSetDevice(nodeRank % count));
    return count;
}

// Exchanges a strip's edge rows with the neighbouring ranks. The strip lives in GPU memory as (rows + 2) x N
// values: ghost row, the strip's rows, ghost row.
class HaloExchange {
public:
    HaloExchange(int N, int rows, heat::mpi::Neighbours nb, bool cudaAware, MPI_Comm comm)
        : N_(N),
          rows_(rows),
          nb_(nb),
          cudaAware_(cudaAware),
          comm_(comm),
          sendUp_(N),
          sendDown_(N),
          recvUp_(N),
          recvDown_(N) {}

    // Fills the ghost rows of `grid` with the neighbours' edge rows. The copies into GPU memory are queued on
    // `stream`, so work queued on `stream` afterwards sees the new ghost rows.
    void exchange(double* grid, cudaStream_t stream) {
        double* first = row(grid, 1);
        double* last = row(grid, rows_);
        double* ghostTop = row(grid, 0);
        double* ghostBottom = row(grid, rows_ + 1);
        if (cudaAware_) {
            send_receive(first, ghostBottom, last, ghostTop);
            return;
        }
        // Host-staged: GPU -> pinned host buffer -> MPI -> pinned host buffer -> GPU.
        copy_row(nb_.up, sendUp_.data(), first, cudaMemcpyDeviceToHost, stream);
        copy_row(nb_.down, sendDown_.data(), last, cudaMemcpyDeviceToHost, stream);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        send_receive(sendUp_.data(), recvDown_.data(), sendDown_.data(), recvUp_.data());
        copy_row(nb_.up, ghostTop, recvUp_.data(), cudaMemcpyHostToDevice, stream);
        copy_row(nb_.down, ghostBottom, recvDown_.data(), cudaMemcpyHostToDevice, stream);
    }

private:
    [[nodiscard]] double* row(double* grid, int r) const { return grid + static_cast<std::size_t>(r) * N_; }

    // Copies one row, unless there is no neighbour on that side.
    void copy_row(int neighbour, double* dst, const double* src, cudaMemcpyKind kind, cudaStream_t stream) const {
        if (neighbour != MPI_PROC_NULL) {
            CHECK_CUDA(cudaMemcpyAsync(dst, src, static_cast<std::size_t>(N_) * sizeof(double), kind, stream));
        }
    }

    // Sends `firstRow` up and `lastRow` down, and receives the neighbours' rows into `fromBelow` and
    // `fromAbove`. At the top and bottom of the grid the neighbour is MPI_PROC_NULL and nothing moves.
    void send_receive(const double* firstRow, double* fromBelow, const double* lastRow, double* fromAbove) const {
        CHECK_MPI(MPI_Sendrecv(firstRow, N_, MPI_DOUBLE, nb_.up, kTagUp, fromBelow, N_, MPI_DOUBLE, nb_.down,
                               kTagUp, comm_, MPI_STATUS_IGNORE));
        CHECK_MPI(MPI_Sendrecv(lastRow, N_, MPI_DOUBLE, nb_.down, kTagDown, fromAbove, N_, MPI_DOUBLE, nb_.up,
                               kTagDown, comm_, MPI_STATUS_IGNORE));
    }

    int N_;
    int rows_;
    heat::mpi::Neighbours nb_;
    bool cudaAware_;
    MPI_Comm comm_;
    PinnedBuffer<double> sendUp_;
    PinnedBuffer<double> sendDown_;
    PinnedBuffer<double> recvUp_;
    PinnedBuffer<double> recvDown_;
};

// This rank's strip in GPU memory, advanced one time step at a time.
class GpuStrip {
public:
    GpuStrip(int N, const heat::Strip& strip, heat::mpi::Neighbours nb, const Settings& settings, MPI_Comm comm)
        : N_(N),
          strip_(strip),
          settings_(settings),
          all_(heat::updated_rows(strip, N)),
          split_(heat::split_rows(strip, N)),
          initial_(heat::initial_strip(strip, N)),
          u_(initial_.size()),
          uNew_(initial_.size()),
          halo_(N, strip.rows, nb, settings.cudaAware, comm) {
        reset();
    }

    // Back to the initial condition.
    void reset() {
        u_.upload(initial_);
        uNew_.upload(initial_);
    }

    void step() {
        using heat::gpu::launch_step;
        using heat::gpu::StepKernel;
        if (settings_.overlap) {
            // The inner rows need no ghost rows: compute them on one stream while the edge rows travel through
            // MPI, then update the edge rows on the other stream once their ghost rows have arrived.
            launch_step(u_.data(), uNew_.data(), N_, split_.inner, heat::kAlpha, StepKernel::Naive,
                        compute_.get());
            halo_.exchange(u_.data(), transfer_.get());
            for (const int r : split_.edges) {
                launch_step(u_.data(), uNew_.data(), N_, {r, r + 1}, heat::kAlpha, StepKernel::Naive,
                            transfer_.get());
            }
        } else {
            halo_.exchange(u_.data(), transfer_.get());
            launch_step(u_.data(), uNew_.data(), N_, all_, heat::kAlpha, StepKernel::Naive, transfer_.get());
        }
        compute_.synchronize();
        transfer_.synchronize();
        swap(u_, uNew_);
    }

    // The strip's rows, without the ghost rows, copied to the host.
    [[nodiscard]] std::vector<double> rows() const { return u_.download(N_, values()); }

    // Sum of the strip's values, computed on the GPU.
    [[nodiscard]] double sum() const { return heat::gpu::sum(u_.data() + N_, values()); }

private:
    [[nodiscard]] std::size_t values() const { return static_cast<std::size_t>(strip_.rows) * N_; }

    int N_;
    heat::Strip strip_;
    Settings settings_;
    heat::RowRange all_;
    heat::RowSplit split_;
    std::vector<double> initial_;
    DeviceBuffer<double> u_;
    DeviceBuffer<double> uNew_;
    Stream compute_;
    Stream transfer_;
    HaloExchange halo_;
};

std::string describe_setup(int processes, int gpus, const Settings& s) {
    return std::to_string(processes) + (processes == 1 ? " process, " : " processes, ") + std::to_string(gpus) +
           (gpus == 1 ? " GPU" : " GPUs") + " per node, " +
           (s.overlap ? "halo exchange overlapped with computation" : "blocking halo exchange") + ", " +
           (s.cudaAware ? "CUDA-aware MPI" : "host-staged transfers");
}

}  // namespace

int main(int argc, char** argv) {
    const heat::mpi::Session mpi(argc, argv);

    heat::Problem problem;
    heat::OutputOptions output;
    Settings settings;
    bool blocking = false;
    heat::ArgParser args("heat_mpi_cuda",
                         "2D heat equation on several GPUs: row strips over MPI processes, one GPU per process.");
    args.add_flag("--no-overlap",
                  "exchange ghost rows first, then compute every row (default: compute the inner rows while the "
                  "ghost rows are exchanged)",
                  &blocking);
    args.add_flag("--cuda-aware",
                  "pass GPU memory straight to MPI instead of copying it through host memory (needs an MPI built "
                  "with CUDA support)",
                  &settings.cudaAware);
    heat::add_output_options(args, output);
    switch (args.parse(argc, argv, problem, mpi.out(), mpi.err())) {
        case heat::ArgParser::Result::Help: return 0;
        case heat::ArgParser::Result::Error: return 1;
        case heat::ArgParser::Result::Ok: break;
    }
    const int N = problem.N;
    if (N < mpi.size()) {
        mpi.err() << "heat_mpi_cuda: N (" << N << ") must be at least the number of processes (" << mpi.size()
                  << ")\n";
        return 1;
    }
    settings.overlap = !blocking;

    const int gpus = select_gpu(MPI_COMM_WORLD);
    const heat::Strip strip = heat::strip_of(mpi.rank(), mpi.size(), N);
    GpuStrip gpu(N, strip, heat::mpi::neighbours(mpi.rank(), mpi.size()), settings, MPI_COMM_WORLD);

    gpu.step();  // untimed warm-up: the first launches also load the kernels onto the GPU
    gpu.reset();

    CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    for (int step = 0; step < problem.K; step++) gpu.step();
    const double localSeconds = MPI_Wtime() - start;

    const double average = heat::mpi::sum_on_root(gpu.sum(), MPI_COMM_WORLD) / (static_cast<double>(N) * N);
    const double seconds = heat::mpi::max_on_root(localSeconds, MPI_COMM_WORLD);

    const std::string csv = output.csv_path(N);
    std::vector<double> grid;
    if (output.verify || !csv.empty()) {
        const std::vector<double> rows = gpu.rows();
        grid = heat::mpi::gather_grid(rows.data(), strip, N, MPI_COMM_WORLD);
    }

    int status = 0;
    if (mpi.rank() == 0) {
        std::ostream& out = mpi.out();
        heat::print_field(out, "Program", "heat_mpi_cuda (MPI + CUDA)");
        heat::print_field(out, "Grid", heat::describe(problem));
        heat::print_field(out, "Setup", describe_setup(mpi.size(), gpus, settings));
        heat::print_field(out, "Device", heat::cuda::current_device().name + " (rank 0)");
        heat::print_timing(out, problem, average, seconds);
        if (output.verify && !heat::verify_against_serial(out, problem, grid, average)) status = 1;
        if (!csv.empty() && !heat::write_csv(out, csv, grid, N)) status = 1;
    }
    CHECK_MPI(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return status;
}
