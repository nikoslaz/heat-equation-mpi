// What the programs print, and the check against the serial solver.
#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include "core/problem.hpp"

namespace heat {

// Prints "key = value" with the keys aligned, e.g. "Average   = 6.01949".
void print_field(std::ostream& os, const std::string& key, const std::string& value);

// A number with 6 significant digits, as std::cout prints it by default.
std::string format_number(double value);

// "256 x 256 grid, 1000 time steps"
std::string describe(const Problem& p);

// The lines every program prints after a run: Average, Time and Rate.
void print_timing(std::ostream& os, const Problem& p, double average, double seconds);

// Largest |a[k] - b[k]|; NaN if any difference is NaN or the sizes differ.
template <typename T>
double max_abs_diff(const std::vector<T>& a, const std::vector<double>& b);

// Tolerance for --verify. A correct result differs from the serial run only by rounding (fused
// multiply-adds on the GPU, or float precision), which stays far below these.
template <typename T>
constexpr double verify_tolerance() {
    return sizeof(T) == sizeof(double) ? 1e-9 : 1e-2;
}

// Compares `grid` and its `average` (as the program computed it, e.g. with a GPU or MPI reduction) with a
// serial run of `p`, prints the "Verify" line and returns whether both match.
template <typename T>
bool verify_against_serial(std::ostream& os, const Problem& p, const std::vector<T>& grid, double average);

// Writes a row-major N x N grid as CSV (one grid row per line), prints the "Output" line and returns
// whether it succeeded.
template <typename T>
bool write_csv(std::ostream& os, const std::string& path, const std::vector<T>& grid, int N);

}  // namespace heat
