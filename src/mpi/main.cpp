#include "heat_mpi.hpp"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: heat K N\n";
        return 1;
    }
    run_mpi(std::stoi(argv[1]), std::stoi(argv[2]));

    return 0;
}
