// Serial CPU version of the scheme, used to check the GPU results point by point.
#pragma once
#include <cstddef>
#include <utility>
#include <vector>

inline std::vector<double> heat_reference(int K, int N, double alpha = 0.2) {
    std::vector<double> u((size_t)N * N, 0.0);
    for (int j = 0; j < N; j++) u[j] = 100.0;
    std::vector<double> uNew = u;

    for (int step = 0; step < K; step++) {
        for (int i = 1; i < N - 1; i++) {
            for (int j = 1; j < N - 1; j++) {
                const size_t idx = (size_t)i * N + j;
                uNew[idx] = u[idx] + alpha * (u[idx - N] + u[idx + N] + u[idx - 1] + u[idx + 1] - 4.0 * u[idx]);
            }
        }
        std::swap(u, uNew);
    }
    return u;
}
