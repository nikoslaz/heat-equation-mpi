#include <algorithm>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>

#include "core/problem.hpp"

namespace {

TEST(InitialGrid, TopRowIsHotEverythingElseZero) {
    const int N = 5;
    const std::vector<float> grid = heat::initial_grid<float>(N);
    ASSERT_EQ(grid.size(), static_cast<std::size_t>(N * N));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) EXPECT_EQ(grid[i * N + j], i == 0 ? 100.0F : 0.0F) << i << "," << j;
    }
}

TEST(Glups, CountsGridPointUpdatesPerSecond) {
    EXPECT_DOUBLE_EQ(heat::glups({1000, 1000}, 1.0), 1.0);
    EXPECT_DOUBLE_EQ(heat::glups({200, 4096}, 0.5), 200.0 * 4096 * 4096 / 0.5 / 1e9);
    EXPECT_EQ(heat::glups({10, 10}, 0.0), 0.0);
}

TEST(StripOf, SplitsRowsEvenlyAndContiguously) {
    for (int N = 3; N <= 40; N++) {
        for (int count = 1; count <= N; count++) {
            int next = 0;
            for (int r = 0; r < count; r++) {
                const heat::Strip s = heat::strip_of(r, count, N);
                EXPECT_EQ(s.first, next) << "N=" << N << " count=" << count << " r=" << r;
                EXPECT_GE(s.rows, N / count);
                EXPECT_LE(s.rows, N / count + 1);
                // The larger strips come first.
                if (r > 0) {
                    EXPECT_LE(s.rows, heat::strip_of(r - 1, count, N).rows);
                }
                next = s.first + s.rows;
            }
            EXPECT_EQ(next, N) << "N=" << N << " count=" << count;
        }
    }
}

TEST(UpdatedRows, SkipsTheGlobalBoundaryRows) {
    // One strip with the whole grid: local rows 2..N-1 are global rows 1..N-2.
    const heat::RowRange whole = heat::updated_rows({0, 10}, 10);
    EXPECT_EQ(whole.begin, 2);
    EXPECT_EQ(whole.end, 10);
    // Top, middle and bottom strips of 10 rows split 4 / 3 / 3.
    const heat::RowRange top = heat::updated_rows({0, 4}, 10);
    EXPECT_EQ(top.begin, 2);
    EXPECT_EQ(top.end, 5);
    const heat::RowRange middle = heat::updated_rows({4, 3}, 10);
    EXPECT_EQ(middle.begin, 1);
    EXPECT_EQ(middle.end, 4);
    const heat::RowRange bottom = heat::updated_rows({7, 3}, 10);
    EXPECT_EQ(bottom.begin, 1);
    EXPECT_EQ(bottom.end, 3);
    // A strip that is only the top boundary row has nothing to update.
    EXPECT_TRUE(heat::updated_rows({0, 1}, 6).empty());
}

TEST(UpdatedRows, EveryInteriorRowIsUpdatedByExactlyOneStrip) {
    for (int N = 3; N <= 30; N++) {
        for (int count = 1; count <= N; count++) {
            std::vector<int> updates(N, 0);
            for (int r = 0; r < count; r++) {
                const heat::Strip s = heat::strip_of(r, count, N);
                const heat::RowRange rows = heat::updated_rows(s, N);
                for (int local = rows.begin; local < rows.end; local++) updates[s.first + local - 1]++;
            }
            for (int g = 0; g < N; g++) {
                const int expected = (g == 0 || g == N - 1) ? 0 : 1;
                EXPECT_EQ(updates[g], expected) << "N=" << N << " count=" << count << " global row " << g;
            }
        }
    }
}

TEST(SplitRows, InnerAndEdgeRowsTogetherAreTheUpdatedRows) {
    for (int N = 3; N <= 30; N++) {
        for (int count = 1; count <= N; count++) {
            for (int r = 0; r < count; r++) {
                const heat::Strip s = heat::strip_of(r, count, N);
                const heat::RowRange all = heat::updated_rows(s, N);
                const heat::RowSplit split = heat::split_rows(s, N);

                std::vector<int> rows;
                for (int local = split.inner.begin; local < split.inner.end; local++) rows.push_back(local);
                for (const int e : split.edges) {
                    EXPECT_TRUE(e == 1 || e == s.rows) << "edge rows are the ones next to a ghost row";
                    rows.push_back(e);
                }
                std::sort(rows.begin(), rows.end());
                std::vector<int> expected;
                for (int local = all.begin; local < all.end; local++) expected.push_back(local);
                EXPECT_EQ(rows, expected) << "N=" << N << " count=" << count << " r=" << r;

                // The inner rows never read a ghost row.
                if (!split.inner.empty()) {
                    EXPECT_GE(split.inner.begin, 2);
                    EXPECT_LE(split.inner.end, s.rows);
                }
            }
        }
    }
}

TEST(InitialStrip, OnlyTheTopStripHoldsTheHotRow) {
    const int N = 4;
    const std::vector<double> top = heat::initial_strip({0, 2}, N);
    ASSERT_EQ(top.size(), static_cast<std::size_t>((2 + 2) * N));
    for (int j = 0; j < N; j++) {
        EXPECT_EQ(top[j], 0.0);        // ghost row
        EXPECT_EQ(top[N + j], 100.0);  // local row 1 = global row 0
        EXPECT_EQ(top[2 * N + j], 0.0);
    }
    const std::vector<double> lower = heat::initial_strip({2, 2}, N);
    EXPECT_TRUE(std::all_of(lower.begin(), lower.end(), [](double v) { return v == 0.0; }));
}

}  // namespace
