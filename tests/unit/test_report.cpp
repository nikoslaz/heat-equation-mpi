#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/report.hpp"
#include "core/stencil.hpp"

namespace {

TEST(PrintField, AlignsTheKeys) {
    std::ostringstream out;
    heat::print_field(out, "Average", "6.01949");
    heat::print_field(out, "Bandwidth", "1 GB/s");
    EXPECT_EQ(out.str(), "Average   = 6.01949\nBandwidth = 1 GB/s\n");
}

TEST(Describe, NamesGridAndSteps) {
    EXPECT_EQ(heat::describe({1000, 256}), "256 x 256 grid, 1000 time steps");
}

TEST(MaxAbsDiff, FindsTheLargestDifference) {
    EXPECT_EQ(heat::max_abs_diff(std::vector<double>{1, 2, 3}, {1, 2, 3}), 0.0);
    EXPECT_DOUBLE_EQ(heat::max_abs_diff(std::vector<double>{1, 2, 3}, {1, 2.5, 2.9}), 0.5);
    EXPECT_FLOAT_EQ(static_cast<float>(heat::max_abs_diff(std::vector<float>{1.5F}, {1.0})), 0.5F);
}

TEST(MaxAbsDiff, ReportsNaNAndSizeMismatchesAsNaN) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(heat::max_abs_diff(std::vector<double>{nan, 1.0, 5.0}, {0.0, 1.0, 0.0})));
    EXPECT_TRUE(std::isnan(heat::max_abs_diff(std::vector<double>{1.0}, {1.0, 2.0})));
}

double average_of(const std::vector<double>& grid) {
    return std::accumulate(grid.begin(), grid.end(), 0.0) / static_cast<double>(grid.size());
}

TEST(VerifyAgainstSerial, AcceptsTheSerialResultAndRejectsAChangedGrid) {
    const heat::Problem p{50, 16};
    std::vector<double> grid = heat::solve_serial(p);
    const double average = average_of(grid);
    std::ostringstream out;
    EXPECT_TRUE(heat::verify_against_serial(out, p, grid, average));
    EXPECT_EQ(out.str().rfind("Verify    = OK", 0), 0U) << out.str();

    grid[5 * 16 + 5] += 1e-6;
    out.str("");
    EXPECT_FALSE(heat::verify_against_serial(out, p, grid, average));
    EXPECT_NE(out.str().find("FAILED"), std::string::npos);
}

TEST(VerifyAgainstSerial, RejectsAWrongAverage) {
    // A correct grid with a wrong average, as a broken reduction would produce.
    const heat::Problem p{50, 16};
    const std::vector<double> grid = heat::solve_serial(p);
    std::ostringstream out;
    EXPECT_FALSE(heat::verify_against_serial(out, p, grid, average_of(grid) * 0.99));
    EXPECT_NE(out.str().find("FAILED"), std::string::npos);
}

TEST(VerifyAgainstSerial, UsesALooserToleranceForFloat) {
    const heat::Problem p{50, 16};
    const std::vector<double> exact = heat::solve_serial(p);
    const std::vector<float> grid(exact.begin(), exact.end());
    std::ostringstream out;
    EXPECT_TRUE(heat::verify_against_serial(out, p, grid, average_of(exact) + 1e-4));
}

TEST(WriteCsv, WritesOneLinePerGridRow) {
    const int N = 3;
    const std::vector<double> grid = {1, 2, 3, 4, 5, 6, 7, 8, 9.5};
    const std::string path = ::testing::TempDir() + "heat_write_csv_test.csv";
    std::ostringstream out;
    ASSERT_TRUE(heat::write_csv(out, path, grid, N));
    EXPECT_EQ(out.str(), "Output    = " + path + "\n");

    const std::ifstream file(path);
    std::stringstream content;
    content << file.rdbuf();
    EXPECT_EQ(content.str(), "1,2,3\n4,5,6\n7,8,9.5\n");
}

TEST(WriteCsv, ReportsAFileThatCannotBeWritten) {
    std::ostringstream out;
    EXPECT_FALSE(heat::write_csv(out, "/nonexistent-directory/grid.csv", std::vector<double>(9, 0.0), 3));
    EXPECT_NE(out.str().find("FAILED"), std::string::npos);
}

}  // namespace
