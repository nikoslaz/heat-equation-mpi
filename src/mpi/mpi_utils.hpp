// MPI helpers shared by the two MPI programs: heat (CPU) and heat_mpi_cuda (GPU).
#pragma once

#include <iosfwd>
#include <vector>

#include <mpi.h>

#include "core/problem.hpp"

// Aborts all processes with MPI's error message, the failed call and its location if an MPI call fails.
#define CHECK_MPI(call)                                                     \
    do {                                                                    \
        const int heat_mpi_status_ = (call);                                \
        if (heat_mpi_status_ != MPI_SUCCESS) {                              \
            ::heat::mpi::fail(heat_mpi_status_, #call, __FILE__, __LINE__); \
        }                                                                   \
    } while (0)

namespace heat::mpi {

[[noreturn]] void fail(int status, const char* call, const char* file, int line);

// Initialises MPI on construction and finalises it on destruction. MPI errors are returned to CHECK_MPI,
// which reports them with context, instead of aborting inside the library.
class Session {
public:
    Session(int& argc, char**& argv);
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    [[nodiscard]] int rank() const { return rank_; }
    [[nodiscard]] int size() const { return size_; }

    // std::cout and std::cerr on rank 0; streams that discard everything on the other ranks.
    [[nodiscard]] std::ostream& out() const;
    [[nodiscard]] std::ostream& err() const;

private:
    int rank_ = 0;
    int size_ = 1;
};

// The ranks holding the strips above and below (MPI_PROC_NULL at the top and bottom of the grid).
struct Neighbours {
    int up = MPI_PROC_NULL;
    int down = MPI_PROC_NULL;
};

Neighbours neighbours(int rank, int size);

// Collects every rank's strip into the full N x N grid on rank 0 (an empty vector on the other ranks).
// `rows` points to the first row of this rank's strip (local row 1).
std::vector<double> gather_grid(const double* rows, const Strip& strip, int N, MPI_Comm comm);

// Sum and maximum over all ranks, valid on rank 0.
double sum_on_root(double value, MPI_Comm comm);
double max_on_root(double value, MPI_Comm comm);

}  // namespace heat::mpi
