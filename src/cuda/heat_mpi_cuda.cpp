// Multi-GPU 2D heat solver: MPI between processes, CUDA inside each process.
// Same row decomposition as heat_mpi.cpp, but each rank keeps its strip (plus two ghost rows)
// in GPU memory. The kernels are in heat_gpu.cu.
#include "cuda_check.hpp"
#include "heat_gpu.hpp"

#include <mpi.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ---- Arguments ----
    int K = 0, N = 0;
    bool overlap = true;     // compute interior rows while the halo exchange runs
    bool cudaAware = false;  // pass GPU pointers straight to MPI (needs a CUDA-aware MPI build)
    bool argsOk = argc >= 3;
    try {
        if (argsOk) {
            K = std::stoi(argv[1]);
            N = std::stoi(argv[2]);
        }
    } catch (...) {
        argsOk = false;
    }
    for (int a = 3; argsOk && a < argc; a++) {
        const std::string arg = argv[a];
        if (arg == "--no-overlap") overlap = false;
        else if (arg == "--cuda-aware") cudaAware = true;
        else argsOk = false;
    }
    if (!argsOk || K < 0 || N < 3) {
        if (rank == 0) std::cout << "Usage: heat_mpi_cuda K N [--no-overlap] [--cuda-aware]\n";
        MPI_Finalize();
        return 1;
    }
    if (N % size != 0) {
        if (rank == 0)
            std::cerr << "Error: N (" << N << ") must be divisible by the number of processes (" << size << ")\n";
        MPI_Finalize();
        return 1;
    }

    // ---- One GPU per rank on each node (ranks share GPUs round-robin if there are fewer) ----
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(nodeRank % deviceCount));

    // ---- Local strip: rows 1..localRows, ghost rows 0 and localRows+1 ----
    const int localRows = N / size;
    const int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    const int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;
    const size_t count = (size_t)(localRows + 2) * N;
    const size_t bytes = count * sizeof(double);
    const size_t rowBytes = (size_t)N * sizeof(double);
    constexpr double alpha = 0.2;

    std::vector<double> h(count, 0.0);
    if (rank == 0)
        for (int j = 0; j < N; j++) h[(size_t)N + j] = 100.0;   // global row 0 is local row 1 of rank 0

    double *dU = nullptr, *dUNew = nullptr;
    CUDA_CHECK(cudaMalloc(&dU, bytes));
    CUDA_CHECK(cudaMalloc(&dUNew, bytes));

    // Pinned host buffers for the host-staged halo exchange (pinned memory allows async copies).
    double *sendUp = nullptr, *sendDown = nullptr, *recvUp = nullptr, *recvDown = nullptr;
    CUDA_CHECK(cudaMallocHost(&sendUp, rowBytes));
    CUDA_CHECK(cudaMallocHost(&sendDown, rowBytes));
    CUDA_CHECK(cudaMallocHost(&recvUp, rowBytes));
    CUDA_CHECK(cudaMallocHost(&recvDown, rowBytes));

    cudaStream_t compute, comm;
    CUDA_CHECK(cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&comm, cudaStreamNonBlocking));

    // Rows this rank updates (local indices, inclusive), as in heat_mpi.cpp.
    const int iStart = (rank == 0) ? 2 : 1;
    const int iEnd = (rank == size - 1) ? localRows - 1 : localRows;
    // Rows 1 and localRows read a ghost row; the rows between them do not.
    const int innerBegin = std::max(iStart, 2);
    const int innerEnd = std::min(iEnd, localRows - 1) + 1;   // exclusive
    std::vector<int> edgeRows;
    for (int r : {1, localRows})
        if (r >= iStart && r <= iEnd && std::find(edgeRows.begin(), edgeRows.end(), r) == edgeRows.end())
            edgeRows.push_back(r);

    // Fill ghost rows 0 and localRows+1 of d with the neighbours' edge rows.
    // Device work is queued on the comm stream.
    auto exchange = [&](double* d) {
        double* firstRow = d + N;
        double* lastRow = d + (size_t)localRows * N;
        double* ghostTop = d;
        double* ghostBottom = d + (size_t)(localRows + 1) * N;

        if (cudaAware) {
            MPI_Sendrecv(firstRow, N, MPI_DOUBLE, up, 0, ghostBottom, N, MPI_DOUBLE, down, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Sendrecv(lastRow, N, MPI_DOUBLE, down, 1, ghostTop, N, MPI_DOUBLE, up, 1,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            return;
        }

        // Host-staged: GPU -> pinned buffer -> MPI -> pinned buffer -> GPU.
        if (up != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(sendUp, firstRow, rowBytes, cudaMemcpyDeviceToHost, comm));
        if (down != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(sendDown, lastRow, rowBytes, cudaMemcpyDeviceToHost, comm));
        CUDA_CHECK(cudaStreamSynchronize(comm));

        MPI_Sendrecv(sendUp, N, MPI_DOUBLE, up, 0, recvDown, N, MPI_DOUBLE, down, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(sendDown, N, MPI_DOUBLE, down, 1, recvUp, N, MPI_DOUBLE, up, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (up != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(ghostTop, recvUp, rowBytes, cudaMemcpyHostToDevice, comm));
        if (down != MPI_PROC_NULL)
            CUDA_CHECK(cudaMemcpyAsync(ghostBottom, recvDown, rowBytes, cudaMemcpyHostToDevice, comm));
    };

    auto step = [&]() {
        if (overlap) {
            // The interior kernel runs on its own stream while the edge rows travel through MPI;
            // the two edge rows are updated once their ghost rows have arrived.
            launch_step_rows(dU, dUNew, N, innerBegin, innerEnd, alpha, StepKernel::Naive, compute);
            exchange(dU);
            for (int r : edgeRows) launch_step_rows(dU, dUNew, N, r, r + 1, alpha, StepKernel::Naive, comm);
        } else {
            exchange(dU);
            launch_step_rows(dU, dUNew, N, iStart, iEnd + 1, alpha, StepKernel::Naive, comm);
        }
        CUDA_CHECK(cudaStreamSynchronize(compute));
        CUDA_CHECK(cudaStreamSynchronize(comm));
        std::swap(dU, dUNew);
    };

    // Untimed warm-up step (loads the kernels onto the GPU), then reset to the initial condition.
    CUDA_CHECK(cudaMemcpy(dU, h.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dUNew, h.data(), bytes, cudaMemcpyHostToDevice));
    step();
    CUDA_CHECK(cudaMemcpy(dU, h.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dUNew, h.data(), bytes, cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int s = 0; s < K; s++) step();
    const double localTime = MPI_Wtime() - t0;

    // ---- Results ----
    const double localSum = gpu_sum(dU + N, (size_t)localRows * N);
    double total = 0.0, maxTime = 0.0;
    MPI_Reduce(&localSum, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (N <= 1024) {
        std::vector<double> mine((size_t)localRows * N);
        CUDA_CHECK(cudaMemcpy(mine.data(), dU + N, mine.size() * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> full;
        if (rank == 0) full.resize((size_t)N * N);
        MPI_Gather(mine.data(), localRows * N, MPI_DOUBLE, full.data(), localRows * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::ofstream f("grid.csv");
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    f << full[(size_t)i * N + j] << (j + 1 < N ? "," : "\n");
        }
    }

    if (rank == 0) {
        std::cout << "Ranks = " << size << ", GPUs on rank 0's node = " << deviceCount
                  << ", overlap = " << (overlap ? "on" : "off")
                  << ", halo transport = " << (cudaAware ? "CUDA-aware MPI" : "host-staged") << "\n";
        std::cout << "Average = " << total / ((double)N * N) << "\n";
        std::cout << "Time = " << maxTime << " s\n";
    }

    CUDA_CHECK(cudaStreamDestroy(compute));
    CUDA_CHECK(cudaStreamDestroy(comm));
    CUDA_CHECK(cudaFreeHost(sendUp));
    CUDA_CHECK(cudaFreeHost(sendDown));
    CUDA_CHECK(cudaFreeHost(recvUp));
    CUDA_CHECK(cudaFreeHost(recvDown));
    CUDA_CHECK(cudaFree(dU));
    CUDA_CHECK(cudaFree(dUNew));

    MPI_Finalize();
    return 0;
}
