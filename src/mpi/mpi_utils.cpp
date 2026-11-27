#include "mpi/mpi_utils.hpp"

#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <streambuf>

namespace heat::mpi {

namespace {

// A stream buffer that throws everything away, for output on ranks other than 0.
class NullBuffer : public std::streambuf {
protected:
    int overflow(int c) override { return c; }
};

std::ostream& null_stream() {
    static NullBuffer buffer;
    static std::ostream stream(&buffer);
    return stream;
}

}  // namespace

void fail(int status, const char* call, const char* file, int line) {
    char message[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(status, message, &length);
    std::fprintf(stderr, "MPI error at %s:%d in %s: %s\n", file, line, call, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();  // not reached
}

Session::Session(int& argc, char**& argv) {
    MPI_Init(&argc, &argv);
    CHECK_MPI(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size_));
}

Session::~Session() {
    MPI_Finalize();
}

std::ostream& Session::out() const {
    return rank_ == 0 ? std::cout : null_stream();
}
std::ostream& Session::err() const {
    return rank_ == 0 ? std::cerr : null_stream();
}

Neighbours neighbours(int rank, int size) {
    return {rank == 0 ? MPI_PROC_NULL : rank - 1, rank == size - 1 ? MPI_PROC_NULL : rank + 1};
}

std::vector<double> gather_grid(const double* rows, const Strip& strip, int N, MPI_Comm comm) {
    int rank = 0;
    int size = 1;
    CHECK_MPI(MPI_Comm_rank(comm, &rank));
    CHECK_MPI(MPI_Comm_size(comm, &size));
    if (static_cast<long long>(N) * N > INT_MAX) {
        if (rank == 0) std::fprintf(stderr, "gather_grid: a %d x %d grid is too large to gather\n", N, N);
        MPI_Abort(comm, EXIT_FAILURE);
    }

    std::vector<double> grid;
    std::vector<int> counts;
    std::vector<int> offsets;
    if (rank == 0) {
        grid.resize(static_cast<std::size_t>(N) * N);
        for (int r = 0; r < size; r++) {
            const Strip s = strip_of(r, size, N);
            counts.push_back(s.rows * N);
            offsets.push_back(s.first * N);
        }
    }
    CHECK_MPI(MPI_Gatherv(rows, strip.rows * N, MPI_DOUBLE, grid.data(), counts.data(), offsets.data(), MPI_DOUBLE,
                          0, comm));
    return grid;
}

double sum_on_root(double value, MPI_Comm comm) {
    double total = 0.0;
    CHECK_MPI(MPI_Reduce(&value, &total, 1, MPI_DOUBLE, MPI_SUM, 0, comm));
    return total;
}

double max_on_root(double value, MPI_Comm comm) {
    double maximum = 0.0;
    CHECK_MPI(MPI_Reduce(&value, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    return maximum;
}

}  // namespace heat::mpi
