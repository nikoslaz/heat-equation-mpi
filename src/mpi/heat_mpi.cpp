#include "heat_mpi.hpp"
#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>

void exchange_ghosts(std::vector<double>& u, int up, int down, int localRows, int N) {
    MPI_Sendrecv(&u[1 * N], N, MPI_DOUBLE, up,   0, &u[(localRows + 1) * N], N, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&u[localRows * N], N, MPI_DOUBLE, down, 0, &u[0], N, MPI_DOUBLE, up,   0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void run_mpi(int K, int N) {
    MPI_Init(nullptr, nullptr);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0) std::cerr << "Error: N (" << N << ") must be divisible by the number of processes (" << size << ")\n";
        MPI_Finalize();
        return;
    }

    int localRows = N / size;
    int up   = (rank == 0)        ? MPI_PROC_NULL : rank - 1;
    int down = (rank == size - 1) ? MPI_PROC_NULL : rank + 1;   

    std::vector<double> u((localRows + 2) * N, 0.0);           
    std::vector<double> uNew((localRows + 2) * N, 0.0);
    constexpr double alpha = 0.2;

    if (rank == 0) {
        for (int j = 0; j < N; j++) u[1 * N + j] = uNew[1 * N + j] = 100.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    for (int step = 0; step < K; step++){
        exchange_ghosts(u, up, down, localRows, N);
        int iStart = (rank == 0)        ? 2 : 1;
        int iEnd   = (rank == size - 1) ? localRows - 1 : localRows;
        for (int i = iStart; i <= iEnd; i++) {
            for (int j = 1; j < N - 1; j++) {
                int idx = i*N + j;
                uNew[idx] = u[idx] + alpha * (u[idx - N] + u[idx + N] + u[idx - 1] + u[idx + 1] - 4*u[idx]);
            }
        }
        std::swap(u, uNew);
    }

    double localTime = MPI_Wtime() - t0;

    double localSum = 0.0;
    for (int i = 1; i <= localRows; i++){
        for (int j = 0; j < N; j++){
            localSum += u[i*N + j];
        }
    }


    double total = 0.0;
    MPI_Reduce(&localSum, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (N <= 1024) {                       
        std::vector<double> full;
        if (rank == 0) full.resize((size_t)N * N);

        MPI_Gather(&u[1 * N], localRows * N, MPI_DOUBLE,     
                full.data(), localRows * N, MPI_DOUBLE,   
                0, MPI_COMM_WORLD);

        if (rank == 0) {
            std::ofstream f("grid.csv");
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    f << full[i*N + j] << (j + 1 < N ? "," : "\n");
        }
    }

    if (rank == 0) {
        std::cout << "Average = " << total / (N * N) << std::endl;
        std::cout << "Time = " << maxTime << " s" << std::endl;
    }

    MPI_Finalize();
}