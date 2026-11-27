#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "core/cli.hpp"

namespace {

using Result = heat::ArgParser::Result;

class ArgParserTest : public ::testing::Test {
protected:
    ArgParserTest() {
        args.add_flag("--verify", "check the result", &verify);
        args.add_int("--tsteps", "S", "steps per launch", &tsteps, 1, 8);
        args.add_choice("--kernel", {"naive", "temporal"}, "which kernel", &kernel);
        args.add_string("--csv", "FILE", "output file", &csv);
    }

    // Parses a command line given as words, without the program name.
    Result parse(std::vector<std::string> words) {
        words.insert(words.begin(), "prog");
        std::vector<const char*> argv;
        argv.reserve(words.size());
        for (const std::string& w : words) argv.push_back(w.c_str());
        out.str("");
        err.str("");
        return args.parse(static_cast<int>(argv.size()), argv.data(), problem, out, err);
    }

    bool verify = false;
    int tsteps = 4;
    std::string kernel = "naive";
    std::string csv;
    heat::Problem problem;
    std::ostringstream out;
    std::ostringstream err;
    heat::ArgParser args{"prog", "A test program."};
};

TEST_F(ArgParserTest, ParsesStepsAndSize) {
    ASSERT_EQ(parse({"100", "256"}), Result::Ok);
    EXPECT_EQ(problem.K, 100);
    EXPECT_EQ(problem.N, 256);
    EXPECT_FALSE(verify);
    EXPECT_EQ(tsteps, 4);
    EXPECT_EQ(kernel, "naive");
    EXPECT_EQ(csv, "");
}

TEST_F(ArgParserTest, AcceptsOptionsBeforeAndAfterTheArguments) {
    ASSERT_EQ(parse({"--verify", "10", "--kernel", "temporal", "20", "--tsteps", "3", "--csv", "out.csv"}),
              Result::Ok);
    EXPECT_EQ(problem.K, 10);
    EXPECT_EQ(problem.N, 20);
    EXPECT_TRUE(verify);
    EXPECT_EQ(tsteps, 3);
    EXPECT_EQ(kernel, "temporal");
    EXPECT_EQ(csv, "out.csv");
}

TEST_F(ArgParserTest, HelpPrintsUsageToOut) {
    EXPECT_EQ(parse({"--help"}), Result::Help);
    EXPECT_NE(out.str().find("Usage: prog K N [options]"), std::string::npos);
    EXPECT_NE(out.str().find("--tsteps S"), std::string::npos);
    EXPECT_NE(out.str().find("--kernel naive|temporal"), std::string::npos);
    EXPECT_NE(out.str().find("(1-8, default 4)"), std::string::npos);
    EXPECT_EQ(err.str(), "");
    EXPECT_EQ(parse({"10", "-h"}), Result::Help);
}

TEST_F(ArgParserTest, RejectsWrongNumberOfArguments) {
    EXPECT_EQ(parse({}), Result::Error);
    EXPECT_EQ(parse({"10"}), Result::Error);
    EXPECT_NE(err.str().find("expected two arguments"), std::string::npos);
    EXPECT_NE(err.str().find("Usage:"), std::string::npos);
    EXPECT_EQ(parse({"10", "20", "30"}), Result::Error);
}

TEST_F(ArgParserTest, RejectsInvalidStepsAndSizes) {
    EXPECT_EQ(parse({"10", "12x"}), Result::Error);
    EXPECT_EQ(parse({"ten", "20"}), Result::Error);
    EXPECT_EQ(parse({"-1", "20"}), Result::Error);
    EXPECT_NE(err.str().find("K must be an integer >= 0"), std::string::npos);
    EXPECT_EQ(parse({"10", "2"}), Result::Error);
    EXPECT_NE(err.str().find("N must be an integer >= 3"), std::string::npos);
    EXPECT_EQ(parse({"10", "99999999999"}), Result::Error);
    EXPECT_EQ(parse({"0", "3"}), Result::Ok);
}

TEST_F(ArgParserTest, RejectsUnknownOptions) {
    EXPECT_EQ(parse({"10", "20", "--bogus"}), Result::Error);
    EXPECT_NE(err.str().find("unknown option '--bogus'"), std::string::npos);
}

TEST_F(ArgParserTest, RejectsBadOptionValues) {
    EXPECT_EQ(parse({"10", "20", "--tsteps", "9"}), Result::Error);
    EXPECT_NE(err.str().find("--tsteps expects an integer in 1-8, got '9'"), std::string::npos);
    EXPECT_EQ(parse({"10", "20", "--tsteps", "0"}), Result::Error);
    EXPECT_EQ(parse({"10", "20", "--tsteps", "two"}), Result::Error);
    EXPECT_EQ(parse({"10", "20", "--kernel", "fast"}), Result::Error);
    EXPECT_NE(err.str().find("--kernel expects one of naive|temporal, got 'fast'"), std::string::npos);
}

TEST_F(ArgParserTest, RejectsAMissingOptionValue) {
    EXPECT_EQ(parse({"10", "20", "--tsteps"}), Result::Error);
    EXPECT_NE(err.str().find("--tsteps needs a value"), std::string::npos);
}

TEST_F(ArgParserTest, WrapsLongHelpText) {
    bool unused = false;
    std::string words(200, 'b');  // four 49-50 character words
    words[50] = words[100] = words[150] = ' ';
    args.add_flag("--long", std::string(30, 'a') + " " + words, &unused);
    std::ostringstream usage;
    args.print_usage(usage);
    std::istringstream lines(usage.str());
    for (std::string line; std::getline(lines, line);) EXPECT_LE(line.size(), 100U) << line;
}

TEST(ParseInt, AcceptsWholeNumbersOnly) {
    int value = 0;
    EXPECT_TRUE(heat::parse_int("42", value));
    EXPECT_EQ(value, 42);
    EXPECT_TRUE(heat::parse_int("-7", value));
    EXPECT_EQ(value, -7);
    EXPECT_FALSE(heat::parse_int("", value));
    EXPECT_FALSE(heat::parse_int("4x", value));
    EXPECT_FALSE(heat::parse_int(" 4", value));
    EXPECT_FALSE(heat::parse_int("2147483648", value));
    EXPECT_EQ(value, -7);  // unchanged after a failure
}

TEST(OutputOptions, WritesSmallGridsToGridCsvByDefault) {
    const heat::OutputOptions options;
    EXPECT_EQ(options.csv_path(256), "grid.csv");
    EXPECT_EQ(options.csv_path(heat::kDefaultCsvMaxSize), "grid.csv");
    EXPECT_EQ(options.csv_path(heat::kDefaultCsvMaxSize + 1), "");
}

TEST(OutputOptions, HonoursCsvAndNoCsv) {
    heat::OutputOptions options;
    options.csv = "big.csv";
    EXPECT_EQ(options.csv_path(4096), "big.csv");
    options.noCsv = true;
    EXPECT_EQ(options.csv_path(256), "");
}

}  // namespace
