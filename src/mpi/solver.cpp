#include "mpi/solver.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>

#include "core/stencil.hpp"
#include "mpi/mpi_utils.hpp"

namespace heat::mpi {

namespace {

constexpr int kTagUp = 0;    // a row travelling to the rank above
constexpr int kTagDown = 1;  // a row travelling to the rank below

double* row(std::vector<double>& u, int r, int N) {
    return u.data() + static_cast<std::size_t>(r) * N;
}

// Starts sending this rank's first row up and its last row down, and receiving the neighbours' edge rows
// into the ghost rows 0 and rows + 1. The ghost rows are ready once all four requests have completed.
std::array<MPI_Request, 4> start_exchange(std::vector<double>& u, int rows, int N, const Neighbours& nb,
                                          MPI_Comm comm) {
    std::array<MPI_Request, 4> requests{};
    MPI_Request* r = requests.data();
    CHECK_MPI(MPI_Irecv(row(u, 0, N), N, MPI_DOUBLE, nb.up, kTagDown, comm, r));
    CHECK_MPI(MPI_Irecv(row(u, rows + 1, N), N, MPI_DOUBLE, nb.down, kTagUp, comm, r + 1));
    CHECK_MPI(MPI_Isend(row(u, 1, N), N, MPI_DOUBLE, nb.up, kTagUp, comm, r + 2));
    CHECK_MPI(MPI_Isend(row(u, rows, N), N, MPI_DOUBLE, nb.down, kTagDown, comm, r + 3));
    return requests;
}

// The same exchange with blocking calls. At the top and bottom of the grid the neighbour is
// MPI_PROC_NULL, so those transfers do nothing and no special case is needed.
void exchange_blocking(std::vector<double>& u, int rows, int N, const Neighbours& nb, MPI_Comm comm) {
    CHECK_MPI(MPI_Sendrecv(row(u, 1, N), N, MPI_DOUBLE, nb.up, kTagUp, row(u, rows + 1, N), N, MPI_DOUBLE, nb.down,
                           kTagUp, comm, MPI_STATUS_IGNORE));
    CHECK_MPI(MPI_Sendrecv(row(u, rows, N), N, MPI_DOUBLE, nb.down, kTagDown, row(u, 0, N), N, MPI_DOUBLE, nb.up,
                           kTagDown, comm, MPI_STATUS_IGNORE));
}

}  // namespace

LocalSolution solve(const Problem& p, const SolverOptions& options, MPI_Comm comm) {
    int rank = 0;
    int size = 1;
    CHECK_MPI(MPI_Comm_rank(comm, &rank));
    CHECK_MPI(MPI_Comm_size(comm, &size));

    const int N = p.N;
    const Strip strip = strip_of(rank, size, N);
    const Neighbours nb = neighbours(rank, size);
    const RowRange all = updated_rows(strip, N);
    const RowSplit split = split_rows(strip, N);

    // Two copies of the strip with its ghost rows (double buffering). The boundary values are set in both
    // and never written again.
    std::vector<double> u = initial_strip(strip, N);
    std::vector<double> uNew = u;

    CHECK_MPI(MPI_Barrier(comm));
    const double start = MPI_Wtime();
    for (int step = 0; step < p.K; step++) {
        if (options.overlap) {
            std::array<MPI_Request, 4> requests = start_exchange(u, strip.rows, N, nb, comm);
            step_rows(u.data(), uNew.data(), N, split.inner);  // needs no ghost rows
            CHECK_MPI(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
            for (const int r : split.edges) step_rows(u.data(), uNew.data(), N, {r, r + 1});
        } else {
            exchange_blocking(u, strip.rows, N, nb, comm);
            step_rows(u.data(), uNew.data(), N, all);
        }
        std::swap(u, uNew);
    }
    const double seconds = MPI_Wtime() - start;

    return {strip, std::move(u), seconds};
}

}  // namespace heat::mpi
