#include "core/cli.hpp"

#include <algorithm>
#include <charconv>
#include <ostream>
#include <system_error>
#include <utility>

namespace heat {

namespace {

std::string join(const std::vector<std::string>& items, const std::string& separator) {
    std::string out;
    for (const std::string& item : items) out += (out.empty() ? "" : separator) + item;
    return out;
}

bool looks_like_option(const std::string& arg) {
    // "-5" is a (negative) number for K or N, not an option.
    return arg.size() > 1 && arg[0] == '-' && (arg[1] < '0' || arg[1] > '9');
}

// Splits `text` into lines of at most `width` characters, breaking at spaces.
std::vector<std::string> wrap(const std::string& text, std::size_t width) {
    std::vector<std::string> lines(1);
    std::size_t start = 0;
    while (start < text.size()) {
        std::size_t end = text.find(' ', start);
        if (end == std::string::npos) end = text.size();
        const std::string word = text.substr(start, end - start);
        if (!lines.back().empty() && lines.back().size() + 1 + word.size() > width) lines.emplace_back();
        lines.back() += (lines.back().empty() ? "" : " ") + word;
        start = end + 1;
    }
    return lines;
}

}  // namespace

bool parse_int(const std::string& text, int& value) {
    const char* first = text.data();
    const char* last = first + text.size();
    int parsed = 0;
    const auto [end, error] = std::from_chars(first, last, parsed);
    if (text.empty() || error != std::errc() || end != last) return false;
    value = parsed;
    return true;
}

ArgParser::ArgParser(std::string program, std::string summary)
    : program_(std::move(program)), summary_(std::move(summary)) {}

void ArgParser::add_flag(const std::string& name, const std::string& help, bool* target) {
    options_.push_back({name, "", help, [target](const std::string&) {
                            *target = true;
                            return std::string();
                        }});
}

void ArgParser::add_int(const std::string& name, const std::string& metavar, const std::string& help, int* target,
                        int min, int max) {
    const std::string range = std::to_string(min) + "-" + std::to_string(max);
    options_.push_back({name, metavar, help + " (" + range + ", default " + std::to_string(*target) + ")",
                        [name, range, target, min, max](const std::string& text) {
                            int value = 0;
                            if (!parse_int(text, value) || value < min || value > max) {
                                return name + " expects an integer in " + range + ", got '" + text + "'";
                            }
                            *target = value;
                            return std::string();
                        }});
}

void ArgParser::add_choice(const std::string& name, const std::vector<std::string>& choices,
                           const std::string& help, std::string* target) {
    const std::string metavar = join(choices, "|");
    options_.push_back({name, metavar, help + " (default " + *target + ")",
                        [name, choices, metavar, target](const std::string& text) {
                            if (std::find(choices.begin(), choices.end(), text) == choices.end()) {
                                return name + " expects one of " + metavar + ", got '" + text + "'";
                            }
                            *target = text;
                            return std::string();
                        }});
}

void ArgParser::add_string(const std::string& name, const std::string& metavar, const std::string& help,
                           std::string* target) {
    options_.push_back({name, metavar, help, [target](const std::string& text) {
                            *target = text;
                            return std::string();
                        }});
}

ArgParser::Result ArgParser::parse(int argc, const char* const argv[], Problem& problem, std::ostream& out,
                                   std::ostream& err) const {
    const auto fail = [&](const std::string& message) {
        err << program_ << ": " << message << "\n\n";
        print_usage(err);
        return Result::Error;
    };

    std::vector<std::string> positional;
    for (int a = 1; a < argc; a++) {
        const std::string arg = argv[a];
        if (arg == "-h" || arg == "--help") {
            print_usage(out);
            return Result::Help;
        }
        if (!looks_like_option(arg)) {
            positional.push_back(arg);
            continue;
        }
        const auto option =
            std::find_if(options_.begin(), options_.end(), [&](const Option& o) { return o.name == arg; });
        if (option == options_.end()) return fail("unknown option '" + arg + "'");
        std::string value;
        if (!option->metavar.empty()) {
            if (a + 1 >= argc) return fail(arg + " needs a value: " + option->metavar);
            value = argv[++a];
        }
        const std::string error = option->apply(value);
        if (!error.empty()) return fail(error);
    }

    if (positional.size() != 2) return fail("expected two arguments, K and N");
    int K = 0;
    int N = 0;
    if (!parse_int(positional[0], K) || K < 0)
        return fail("K must be an integer >= 0, got '" + positional[0] + "'");
    if (!parse_int(positional[1], N) || N < 3)
        return fail("N must be an integer >= 3, got '" + positional[1] + "'");
    problem = {K, N};
    return Result::Ok;
}

void ArgParser::print_usage(std::ostream& os) const {
    std::vector<std::pair<std::string, std::string>> rows = {{"K", "number of time steps (>= 0)"},
                                                             {"N", "grid size: N x N points (>= 3)"}};
    const std::size_t firstOption = rows.size();
    for (const Option& o : options_)
        rows.emplace_back(o.name + (o.metavar.empty() ? "" : " " + o.metavar), o.help);
    rows.emplace_back("-h, --help", "show this help");

    std::size_t width = 0;
    for (const auto& row : rows) width = std::max(width, row.first.size());
    constexpr std::size_t kLineWidth = 100;
    const std::size_t helpWidth = kLineWidth > width + 40 ? kLineWidth - width - 4 : 36;

    os << "Usage: " << program_ << " K N [options]\n" << summary_ << "\n";
    for (std::size_t r = 0; r < rows.size(); r++) {
        if (r == 0) os << "\nArguments:\n";
        if (r == firstOption) os << "\nOptions:\n";
        const std::vector<std::string> help = wrap(rows[r].second, helpWidth);
        for (std::size_t line = 0; line < help.size(); line++) {
            const std::string& name = line == 0 ? rows[r].first : std::string();
            os << "  " << name << std::string(width - name.size() + 2, ' ') << help[line] << "\n";
        }
    }
}

std::string OutputOptions::csv_path(int N) const {
    if (noCsv) return "";
    if (!csv.empty()) return csv;
    return N <= kDefaultCsvMaxSize ? "grid.csv" : "";
}

void add_output_options(ArgParser& args, OutputOptions& options) {
    args.add_flag("--verify", "compare every grid point with a serial CPU run; exit with 1 if they differ",
                  &options.verify);
    args.add_string("--csv", "FILE", "write the final grid to FILE (default: grid.csv when N <= 1024)",
                    &options.csv);
    args.add_flag("--no-csv", "do not write the final grid", &options.noCsv);
}

}  // namespace heat
