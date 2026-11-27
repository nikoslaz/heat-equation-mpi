// The problem all three programs solve, and the row decomposition shared by the MPI versions.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace heat {

// Diffusion number alpha = D * dt / dx^2. The explicit scheme is stable for alpha <= 0.25.
inline constexpr double kAlpha = 0.2;

// The top edge is held at this temperature; the other three edges are held at 0.
inline constexpr double kTopTemperature = 100.0;

// K explicit time steps on an N x N grid (named as on the command line: heat K N).
struct Problem {
    int K = 0;
    int N = 0;
};

// The initial condition as a row-major N x N grid: top row hot, everything else 0.
template <typename T>
std::vector<T> initial_grid(int N) {
    std::vector<T> grid(static_cast<std::size_t>(N) * N, T(0));
    std::fill_n(grid.begin(), N, static_cast<T>(kTopTemperature));
    return grid;
}

// Grid points updated per second, in billions (GLUP/s: giga lattice updates per second).
inline double glups(const Problem& p, double seconds) {
    return seconds > 0 ? static_cast<double>(p.N) * p.N * p.K / seconds / 1e9 : 0.0;
}

// ---- Row decomposition ----------------------------------------------------------------------------------

// Rows [first, first + rows) of the global grid, owned by one process.
struct Strip {
    int first = 0;
    int rows = 0;
};

// Splits N rows over `count` processes as evenly as possible: the first N % count processes get one row
// more than the others.
inline Strip strip_of(int index, int count, int N) {
    const int base = N / count;
    const int extra = N % count;
    return {index * base + std::min(index, extra), base + (index < extra ? 1 : 0)};
}

// A half-open range of rows [begin, end).
struct RowRange {
    int begin = 0;
    int end = 0;

    [[nodiscard]] bool empty() const { return end <= begin; }
    [[nodiscard]] int size() const { return empty() ? 0 : end - begin; }
};

// A strip is stored with one ghost row above and one below it, so local row r (1 <= r <= rows) is global
// row first + r - 1. Returns the local rows the stencil updates: all of them except global rows 0 and N - 1,
// which are part of the fixed boundary.
inline RowRange updated_rows(const Strip& s, int N) {
    return {std::max(1, 2 - s.first), std::min(s.rows, N - 1 - s.first) + 1};
}

// The initial condition of one strip, stored with its two ghost rows: (rows + 2) x N values.
inline std::vector<double> initial_strip(const Strip& s, int N) {
    std::vector<double> grid(static_cast<std::size_t>(s.rows + 2) * N, 0.0);
    if (s.first == 0) std::fill_n(grid.begin() + N, N, kTopTemperature);  // global row 0 is local row 1
    return grid;
}

// The updated rows, split into the inner rows, which do not read a ghost row, and the edge rows (local rows
// 1 and `rows`), which do. The inner rows can be computed while the ghost rows are still in transit.
struct RowSplit {
    RowRange inner;
    std::vector<int> edges;
};

inline RowSplit split_rows(const Strip& s, int N) {
    const RowRange all = updated_rows(s, N);
    RowSplit split{{std::max(all.begin, 2), std::min(all.end, s.rows)}, {}};
    for (const int r : {1, s.rows}) {
        const bool updated = r >= all.begin && r < all.end;
        const bool seen = !split.edges.empty() && split.edges.back() == r;
        if (updated && !seen) split.edges.push_back(r);
    }
    return split;
}

}  // namespace heat
