cmake_minimum_required(VERSION 3.16)
project(heat2d LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(MPI REQUIRED)
add_executable(heat main.cpp heat_mpi.cpp)
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
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  add_executable(heat_cuda main.cpp heat_cuda.cu)
  target_compile_definitions(heat_cuda PRIVATE HEAT_USE_CUDA)
  target_compile_options(heat_cuda PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)

  add_custom_target(run_cuda
    COMMAND $<TARGET_FILE:heat_cuda> ${HEAT_K} ${HEAT_N}
    DEPENDS heat_cuda
    USES_TERMINAL
  )
else()
  message(STATUS "No CUDA compiler found: building the MPI version only")
endif()
