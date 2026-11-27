// heat: the 2D heat equation on the CPU, split into row strips over MPI processes.
#include <cstddef>
#include <numeric>
#include <ostream>
#include <string>
#include <vector>

#include "core/cli.hpp"
#include "core/report.hpp"
#include "mpi/mpi_utils.hpp"
#include "mpi/solver.hpp"

int main(int argc, char** argv) {
    const heat::mpi::Session mpi(argc, argv);

    heat::Problem problem;
    heat::OutputOptions output;
    bool blocking = false;
    heat::ArgParser args("heat", "2D heat equation on the CPU, split into row strips over MPI processes.");
    args.add_flag("--no-overlap",
                  "exchange ghost rows with blocking MPI_Sendrecv, then compute (default: non-blocking "
                  "exchange, overlapped with computing the inner rows)",
                  &blocking);
    heat::add_output_options(args, output);
    switch (args.parse(argc, argv, problem, mpi.out(), mpi.err())) {
        case heat::ArgParser::Result::Help: return 0;
        case heat::ArgParser::Result::Error: return 1;
        case heat::ArgParser::Result::Ok: break;
    }
    const int N = problem.N;
    if (N < mpi.size()) {
        mpi.err() << "heat: N (" << N << ") must be at least the number of processes (" << mpi.size() << ")\n";
        return 1;
    }

    const heat::mpi::LocalSolution local = heat::mpi::solve(problem, {!blocking}, MPI_COMM_WORLD);

    const double* rows = local.rows(N);
    const double localSum = std::accumulate(rows, rows + static_cast<std::size_t>(local.strip.rows) * N, 0.0);
    const double average = heat::mpi::sum_on_root(localSum, MPI_COMM_WORLD) / (static_cast<double>(N) * N);
    const double seconds = heat::mpi::max_on_root(local.seconds, MPI_COMM_WORLD);

    const std::string csv = output.csv_path(N);
    std::vector<double> grid;
    if (output.verify || !csv.empty()) grid = heat::mpi::gather_grid(rows, local.strip, N, MPI_COMM_WORLD);

    int status = 0;
    if (mpi.rank() == 0) {
        std::ostream& out = mpi.out();
        const std::string processes = std::to_string(mpi.size()) + (mpi.size() == 1 ? " process" : " processes");
        heat::print_field(out, "Program", "heat (CPU, MPI)");
        heat::print_field(out, "Grid", heat::describe(problem));
        heat::print_field(
            out, "Setup",
            processes + ", " +
                (blocking ? "blocking halo exchange" : "non-blocking halo exchange overlapped with computation"));
        heat::print_timing(out, problem, average, seconds);
        if (output.verify && !heat::verify_against_serial(out, problem, grid, average)) status = 1;
        if (!csv.empty() && !heat::write_csv(out, csv, grid, N)) status = 1;
    }
    CHECK_MPI(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    return status;
}
