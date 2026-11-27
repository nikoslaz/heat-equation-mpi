cmake_minimum_required(VERSION 3.22)
project(heat2d
  VERSION 1.0.0
  DESCRIPTION "2D heat equation solver: MPI on CPUs, CUDA on one GPU, MPI + CUDA on several GPUs"
  HOMEPAGE_URL "https://github.com/nikoslaz/heat-equation-mpi"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(HEAT_BUILD_TESTS "Build the tests (GoogleTest is downloaded if it is not installed)" ON)
option(HEAT_WERROR "Treat compiler warnings as errors" OFF)
option(HEAT_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
set(HEAT_CUDA AUTO CACHE STRING
  "CUDA programs: AUTO (when nvcc is found), ON, OFF, or EMULATOR (build them against a CPU emulator of CUDA)")
set_property(CACHE HEAT_CUDA PROPERTY STRINGS AUTO ON OFF EMULATOR)

find_package(MPI REQUIRED COMPONENTS CXX)

# ---- Compiler options for this project's own targets ----------------------------------------------------
add_library(heat_options INTERFACE)
target_compile_options(heat_options INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow>
  $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<BOOL:${HEAT_WERROR}>>:-Werror>)
if(HEAT_SANITIZE)
  target_compile_options(heat_options INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined -fno-sanitize-recover=undefined -fno-omit-frame-pointer>)
  target_link_options(heat_options INTERFACE -fsanitize=address,undefined)
endif()

# ---- Code shared by all programs (src/core) and by the MPI programs (src/mpi/mpi_utils) -----------------
add_library(heat_core STATIC src/core/cli.cpp src/core/report.cpp)
target_include_directories(heat_core PUBLIC src)
target_link_libraries(heat_core PUBLIC heat_options)

add_library(heat_mpi_utils STATIC src/mpi/mpi_utils.cpp)
target_link_libraries(heat_mpi_utils PUBLIC heat_core MPI::MPI_CXX)

# ---- heat: CPU, MPI ----------------------------------------------------------------------------------------
add_executable(heat src/mpi/main.cpp src/mpi/solver.cpp)
target_link_libraries(heat PRIVATE heat_mpi_utils)

# ---- heat_cuda and heat_mpi_cuda: one GPU, and MPI + CUDA -------------------------------------------------
include(cmake/cuda.cmake)

# ---- Convenience targets: cmake --build <dir> --target run (or run_cuda) ------------------------------------
set(HEAT_NP 4 CACHE STRING "MPI processes for the run targets")
set(HEAT_K 1000 CACHE STRING "Time steps for the run targets")
set(HEAT_N 256 CACHE STRING "Grid size for the run targets")
add_custom_target(run
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${HEAT_NP} $<TARGET_FILE:heat> ${HEAT_K} ${HEAT_N}
  DEPENDS heat
  USES_TERMINAL)
if(TARGET heat_cuda)
  add_custom_target(run_cuda
    COMMAND $<TARGET_FILE:heat_cuda> ${HEAT_K} ${HEAT_N} --kernel temporal
    DEPENDS heat_cuda
    USES_TERMINAL)
endif()

# ---- Tests ---------------------------------------------------------------------------------------------------
if(HEAT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
