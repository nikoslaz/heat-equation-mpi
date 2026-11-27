#include "core/report.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <numeric>
#include <ostream>
#include <sstream>

#include "core/stencil.hpp"

namespace heat {

namespace {
constexpr std::size_t kKeyWidth = 9;  // length of the longest key, "Bandwidth"
}

void print_field(std::ostream& os, const std::string& key, const std::string& value) {
    const std::size_t pad = key.size() < kKeyWidth ? kKeyWidth - key.size() : 0;
    os << key << std::string(pad, ' ') << " = " << value << '\n';
}

std::string format_number(double value) {
    std::ostringstream s;
    s << value;
    return s.str();
}

std::string describe(const Problem& p) {
    return std::to_string(p.N) + " x " + std::to_string(p.N) + " grid, " + std::to_string(p.K) + " time steps";
}

void print_timing(std::ostream& os, const Problem& p, double average, double seconds) {
    print_field(os, "Average", format_number(average));
    print_field(os, "Time", format_number(seconds) + " s");
    print_field(os, "Rate", format_number(glups(p, seconds)) + " GLUP/s (billion grid-point updates per second)");
}

template <typename T>
double max_abs_diff(const std::vector<T>& a, const std::vector<double>& b) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    if (a.size() != b.size()) return kNaN;
    double worst = 0.0;
    for (std::size_t k = 0; k < a.size(); k++) {
        const double d = std::fabs(static_cast<double>(a[k]) - b[k]);
        if (std::isnan(d)) return kNaN;
        worst = std::max(worst, d);
    }
    return worst;
}

template <typename T>
bool verify_against_serial(std::ostream& os, const Problem& p, const std::vector<T>& grid, double average) {
    const std::vector<double> serial = solve_serial(p);
    const double serialAverage =
        std::accumulate(serial.begin(), serial.end(), 0.0) / static_cast<double>(serial.size());
    const double gridDiff = max_abs_diff(grid, serial);
    const double averageDiff = std::fabs(average - serialAverage);
    const double tolerance = verify_tolerance<T>();
    const bool ok = gridDiff <= tolerance && averageDiff <= tolerance;
    print_field(os, "Verify",
                std::string(ok ? "OK" : "FAILED") +
                    ", largest difference from a serial run: " + format_number(gridDiff) + " (grid), " +
                    format_number(averageDiff) + " (average); tolerance " + format_number(tolerance));
    return ok;
}

template <typename T>
bool write_csv(std::ostream& os, const std::string& path, const std::vector<T>& grid, int N) {
    std::ofstream file(path);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) file << grid[static_cast<std::size_t>(i) * N + j] << (j + 1 < N ? ',' : '\n');
    }
    if (!file) {
        print_field(os, "Output", "FAILED to write " + path);
        return false;
    }
    print_field(os, "Output", path);
    return true;
}

template double max_abs_diff<float>(const std::vector<float>&, const std::vector<double>&);
template double max_abs_diff<double>(const std::vector<double>&, const std::vector<double>&);
template bool verify_against_serial<float>(std::ostream&, const Problem&, const std::vector<float>&, double);
template bool verify_against_serial<double>(std::ostream&, const Problem&, const std::vector<double>&, double);
template bool write_csv<float>(std::ostream&, const std::string&, const std::vector<float>&, int);
template bool write_csv<double>(std::ostream&, const std::string&, const std::vector<double>&, int);

}  // namespace heat
