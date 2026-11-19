#ifdef HEAT_USE_CUDA
#include "heat_cuda.hpp"
#else
#include "heat_mpi.hpp"
#endif
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 3) {
#ifdef HEAT_USE_CUDA
        std::cout << "Usage: heat_cuda K N\n";
#else
        std::cout << "Usage: heat K N\n";
#endif
        return 1;
    }
#ifdef HEAT_USE_CUDA
    run_cuda(std::stoi(argv[1]), std::stoi(argv[2]));
#else
    run_mpi(std::stoi(argv[1]), std::stoi(argv[2]));
#endif

    return 0;
}
