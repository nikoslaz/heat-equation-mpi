// The CPU solver: the grid is split into row strips, one per MPI process, and neighbouring processes
// exchange their edge rows ("ghost rows") every time step.
#pragma once

#include <cstddef>
#include <vector>

#include <mpi.h>

#include "core/problem.hpp"

namespace heat::mpi {

struct SolverOptions {
    // Exchange ghost rows with non-blocking MPI_Isend/MPI_Irecv and compute the inner rows meanwhile.
    // When false, exchange with blocking MPI_Sendrecv first and then compute every row.
    bool overlap = true;
};

// This rank's part of the solution.
struct LocalSolution {
    Strip strip;
    std::vector<double> grid;  // (strip.rows + 2) x N values: ghost row, the strip's rows, ghost row
    double seconds = 0.0;      // duration of this rank's time-step loop

    // The first row of the strip (local row 1); the strip's strip.rows * N values follow it.
    [[nodiscard]] const double* rows(int N) const { return grid.data() + N; }
};

LocalSolution solve(const Problem& p, const SolverOptions& options, MPI_Comm comm);

}  // namespace heat::mpi
