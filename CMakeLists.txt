cmake_minimum_required(VERSION 3.16)
project(heat2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED)
add_executable(heat src/mpi/main.cpp src/mpi/heat_mpi.cpp)
target_compile_options(heat PRIVATE -Wall -Wextra)
target_link_libraries(heat PRIVATE MPI::MPI_CXX)

set(NP 4 CACHE STRING "Number of MPI processes")
set(HEAT_K 1000 CACHE STRING "Time steps")
set(HEAT_N 256 CACHE STRING "Grid size")

add_custom_target(run
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} $<TARGET_FILE:heat> ${HEAT_K} ${HEAT_N}
  DEPENDS heat
  USES_TERMINAL
)

# Optional CUDA build: only when a CUDA compiler (nvcc) is found.
#   heat_cuda      single GPU (kernels: naive, shared, temporal; double or float)
#   heat_mpi_cuda  MPI + CUDA, one GPU per rank
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  # Compile for the GPU in this machine unless an architecture is given (e.g. -DCMAKE_CUDA_ARCHITECTURES=75).
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # Kernels; the programs that use them are plain C++.
  add_library(heat_gpu STATIC src/cuda/heat_gpu.cu)
  target_compile_options(heat_gpu PRIVATE -lineinfo)   # source lines in Nsight Compute profiles
  target_link_libraries(heat_gpu PUBLIC CUDA::cudart)

  add_executable(heat_cuda src/cuda/heat_cuda.cpp)
  target_compile_options(heat_cuda PRIVATE -Wall -Wextra)
  target_link_libraries(heat_cuda PRIVATE heat_gpu)

  add_executable(heat_mpi_cuda src/cuda/heat_mpi_cuda.cpp)
  target_compile_options(heat_mpi_cuda PRIVATE -Wall -Wextra)
  target_link_libraries(heat_mpi_cuda PRIVATE heat_gpu MPI::MPI_CXX)

  set(HEAT_CUDA_KERNEL naive CACHE STRING "Kernel for the run_cuda target: naive, shared or temporal")
  add_custom_target(run_cuda
    COMMAND $<TARGET_FILE:heat_cuda> ${HEAT_K} ${HEAT_N} --kernel ${HEAT_CUDA_KERNEL}
    DEPENDS heat_cuda
    USES_TERMINAL
  )
  add_custom_target(run_mpi_cuda
    COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${NP} $<TARGET_FILE:heat_mpi_cuda> ${HEAT_K} ${HEAT_N}
    DEPENDS heat_mpi_cuda
    USES_TERMINAL
  )
else()
  message(STATUS "No CUDA compiler found: building the MPI version only")
endif()
