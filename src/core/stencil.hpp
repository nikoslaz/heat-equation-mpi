// The 5-point stencil on the CPU, and the serial solver the parallel versions are checked against.
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/problem.hpp"

namespace heat {

// One explicit time step on rows [rows.begin, rows.end), columns 1..N-2, of a row-major array with N columns:
//   uNew = u + alpha * (u_up + u_down + u_left + u_right - 4 u)
// Rows rows.begin - 1 and rows.end are read (boundary or ghost rows) but not written.
inline void step_rows(const double* u, double* uNew, int N, RowRange rows, double alpha = kAlpha) {
    for (int i = rows.begin; i < rows.end; i++) {
        const double* up = u + static_cast<std::size_t>(i - 1) * N;
        const double* mid = up + N;
        const double* down = mid + N;
        double* out = uNew + static_cast<std::size_t>(i) * N;
        for (int j = 1; j < N - 1; j++) {
            out[j] = mid[j] + alpha * (up[j] + down[j] + mid[j - 1] + mid[j + 1] - 4.0 * mid[j]);
        }
    }
}

// The whole grid after p.K time steps, computed serially.
inline std::vector<double> solve_serial(const Problem& p) {
    std::vector<double> u = initial_grid<double>(p.N);
    std::vector<double> uNew = u;
    for (int step = 0; step < p.K; step++) {
        step_rows(u.data(), uNew.data(), p.N, {1, p.N - 1});
        std::swap(u, uNew);
    }
    return u;
}

}  // namespace heat
