#include <algorithm>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>

#include "core/stencil.hpp"

namespace {

double at(const std::vector<double>& u, int N, int i, int j) {
    return u[static_cast<std::size_t>(i) * N + j];
}

TEST(Stencil, FirstTwoStepsByHand) {
    // 4 x 4 grid: the interior is the 2 x 2 block (1..2, 1..2).
    // Step 1: u(1,j) = 0 + 0.2 * (100 + 0 + 0 + 0 - 0) = 20, row 2 stays 0.
    // Step 2: u(1,1) = 20 + 0.2 * (100 + 0 + 0 + 20 - 80) = 28, u(2,1) = 0 + 0.2 * 20 = 4.
    const int N = 4;
    const std::vector<double> one = heat::solve_serial({1, N});
    EXPECT_DOUBLE_EQ(at(one, N, 1, 1), 20.0);
    EXPECT_DOUBLE_EQ(at(one, N, 1, 2), 20.0);
    EXPECT_DOUBLE_EQ(at(one, N, 2, 1), 0.0);

    const std::vector<double> two = heat::solve_serial({2, N});
    EXPECT_DOUBLE_EQ(at(two, N, 1, 1), 28.0);
    EXPECT_DOUBLE_EQ(at(two, N, 1, 2), 28.0);
    EXPECT_DOUBLE_EQ(at(two, N, 2, 1), 4.0);
    EXPECT_DOUBLE_EQ(at(two, N, 2, 2), 4.0);
}

TEST(Stencil, BoundaryStaysFixed) {
    const int N = 12;
    const std::vector<double> u = heat::solve_serial({300, N});
    for (int k = 0; k < N; k++) {
        EXPECT_EQ(at(u, N, 0, k), 100.0);
        EXPECT_EQ(at(u, N, N - 1, k), 0.0);
        EXPECT_EQ(at(u, N, k, 0), k == 0 ? 100.0 : 0.0);
        EXPECT_EQ(at(u, N, k, N - 1), k == 0 ? 100.0 : 0.0);
    }
}

TEST(Stencil, StaysBetweenTheBoundaryTemperatures) {
    // With alpha <= 0.25 each new value is a weighted average of old ones (maximum principle).
    const std::vector<double> u = heat::solve_serial({500, 20});
    EXPECT_GE(*std::min_element(u.begin(), u.end()), 0.0);
    EXPECT_LE(*std::max_element(u.begin(), u.end()), 100.0);
}

TEST(Stencil, IsSymmetricLeftToRight) {
    const int N = 15;
    const std::vector<double> u = heat::solve_serial({200, N});
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) EXPECT_NEAR(at(u, N, i, j), at(u, N, i, N - 1 - j), 1e-12);
    }
}

TEST(Stencil, UpdatingRowsInPiecesEqualsUpdatingThemAtOnce) {
    const int N = 9;
    std::vector<double> u = heat::solve_serial({7, N});  // something non-trivial
    std::vector<double> whole(u.size(), -1.0);
    std::vector<double> pieces(u.size(), -1.0);
    heat::step_rows(u.data(), whole.data(), N, {1, N - 1});
    heat::step_rows(u.data(), pieces.data(), N, {1, 3});
    heat::step_rows(u.data(), pieces.data(), N, {3, 4});
    heat::step_rows(u.data(), pieces.data(), N, {4, N - 1});
    EXPECT_EQ(whole, pieces);
}

TEST(Stencil, KnownAverageFor256x256After1000Steps) {
    // The reference value quoted in the README, which every program reproduces.
    const std::vector<double> u = heat::solve_serial({1000, 256});
    const double average = std::accumulate(u.begin(), u.end(), 0.0) / static_cast<double>(u.size());
    EXPECT_NEAR(average, 6.01949, 5e-6);
}

}  // namespace
