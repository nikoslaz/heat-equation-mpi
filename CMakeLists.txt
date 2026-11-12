cmake_minimum_required(VERSION 3.16)
project(heat2d CXX)

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