// A small command-line parser shared by the programs, which all take "K N [options]".
#pragma once

#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

#include "core/problem.hpp"

namespace heat {

class ArgParser {
public:
    enum class Result { Ok, Help, Error };

    ArgParser(std::string program, std::string summary);

    // --name: sets *target to true.
    void add_flag(const std::string& name, const std::string& help, bool* target);
    // --name VALUE: an integer in [min, max].
    void add_int(const std::string& name, const std::string& metavar, const std::string& help, int* target,
                 int min, int max);
    // --name VALUE: one of `choices`.
    void add_choice(const std::string& name, const std::vector<std::string>& choices, const std::string& help,
                    std::string* target);
    // --name VALUE: any text.
    void add_string(const std::string& name, const std::string& metavar, const std::string& help,
                    std::string* target);

    // Parses the command line into `problem` and the registered targets. On Help the usage is written to
    // `out`; on Error a message and the usage are written to `err`.
    Result parse(int argc, const char* const argv[], Problem& problem, std::ostream& out, std::ostream& err) const;

    void print_usage(std::ostream& os) const;

private:
    struct Option {
        std::string name;
        std::string metavar;  // empty for a flag
        std::string help;
        std::function<std::string(const std::string&)> apply;  // returns an error message, "" on success
    };

    std::string program_;
    std::string summary_;
    std::vector<Option> options_;
};

// Strict integer parsing: the whole text must be a number that fits in an int.
bool parse_int(const std::string& text, int& value);

// Grids up to this size are written to grid.csv unless told otherwise.
inline constexpr int kDefaultCsvMaxSize = 1024;

// Options shared by all programs.
struct OutputOptions {
    bool verify = false;  // --verify
    bool noCsv = false;   // --no-csv
    std::string csv;      // --csv FILE

    // Where to write the final grid, or "" for nowhere.
    [[nodiscard]] std::string csv_path(int N) const;
};

void add_output_options(ArgParser& args, OutputOptions& options);

}  // namespace heat
