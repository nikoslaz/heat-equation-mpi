# ---- Unit tests (GoogleTest) ------------------------------------------------------------------------------
find_package(GTest 1.11 CONFIG QUIET)
if(NOT GTest_FOUND)
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)  # timestamps of extracted files = extraction time
  endif()
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c)
  set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(heat_unit_tests
  unit/test_cli.cpp
  unit/test_problem.cpp
  unit/test_report.cpp
  unit/test_stencil.cpp)
target_link_libraries(heat_unit_tests PRIVATE heat_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(heat_unit_tests TEST_PREFIX "unit." PROPERTIES LABELS unit)

# ---- Program tests ----------------------------------------------------------------------------------------
# Each test runs one of the programs. With --verify the program compares every grid point with a serial run
# and prints FAILED (and exits with 1) on a mismatch; tests with EXPECT also check the printed average.
# Labels: cpu, cuda (CUDA code, real GPU or emulator), gpu (needs a real GPU), mpi (runs under mpiexec).

# Tests start up to 7 processes. Open MPI refuses more processes than cores unless told --oversubscribe
# (other MPIs do not know the flag), so ask mpiexec which MPI it is.
set(heat_mpiexec_flags ${MPIEXEC_PREFLAGS})
execute_process(COMMAND ${MPIEXEC_EXECUTABLE} --version OUTPUT_VARIABLE mpiexec_version ERROR_VARIABLE mpiexec_version
                RESULT_VARIABLE mpiexec_result)
if(mpiexec_version MATCHES "Open MPI|OpenRTE|prterun")
  list(APPEND heat_mpiexec_flags --oversubscribe)
endif()

# heat_add_test(<name> TARGET <target> [NP <processes>] [EXPECT <average>] [LABELS <label>...] ARGS <arg>...)
function(heat_add_test name)
  cmake_parse_arguments(PARSE_ARGV 1 T "" "TARGET;NP;EXPECT" "LABELS;ARGS")
  if(T_NP)
    set(command ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${T_NP} ${heat_mpiexec_flags}
                $<TARGET_FILE:${T_TARGET}> ${MPIEXEC_POSTFLAGS} ${T_ARGS})
  else()
    set(command $<TARGET_FILE:${T_TARGET}> ${T_ARGS})
    set(T_NP 1)
  endif()
  add_test(NAME ${name} COMMAND ${command})
  set_tests_properties(${name} PROPERTIES
    LABELS "${T_LABELS}"
    PROCESSORS ${T_NP}
    TIMEOUT 300
    FAIL_REGULAR_EXPRESSION "FAILED")
  if(T_EXPECT)
    string(REPLACE "." "\\." expected "${T_EXPECT}")
    set_tests_properties(${name} PROPERTIES PASS_REGULAR_EXPRESSION "Average += ${expected}")
  endif()
endfunction()

# ---- heat (CPU) ----
heat_add_test(heat.help TARGET heat NP 1 LABELS cpu mpi ARGS --help)
set_tests_properties(heat.help PROPERTIES PASS_REGULAR_EXPRESSION "Usage: heat K N")
heat_add_test(heat.rejects_bad_arguments TARGET heat NP 1 LABELS cpu mpi ARGS 10 2x)
set_tests_properties(heat.rejects_bad_arguments PROPERTIES PASS_REGULAR_EXPRESSION "N must be an integer")

foreach(np 1 2 3 4)
  heat_add_test(heat.np${np}.overlap TARGET heat NP ${np} EXPECT 6.01949 LABELS cpu mpi
    ARGS 1000 256 --verify --no-csv)
  heat_add_test(heat.np${np}.blocking TARGET heat NP ${np} EXPECT 6.01949 LABELS cpu mpi
    ARGS 1000 256 --verify --no-csv --no-overlap)
endforeach()
# Uneven strips, and enough steps for the heat to reach every row.
heat_add_test(heat.np3.uneven TARGET heat NP 3 LABELS cpu mpi ARGS 600 50 --verify --no-csv)
heat_add_test(heat.np7.uneven TARGET heat NP 7 LABELS cpu mpi ARGS 600 50 --verify --no-csv)
# Strips of one row each: every row is an edge row.
heat_add_test(heat.np6.one_row_strips TARGET heat NP 6 LABELS cpu mpi ARGS 50 6 --verify --no-csv)

# ---- heat_cuda and heat_mpi_cuda (on a GPU, or on the CPU emulator) ----
if(TARGET heat_cuda)
  foreach(variant naive shared temporal1 temporal3 temporal8)
    if(variant MATCHES "^temporal([0-9]+)$")
      set(kernel_args --kernel temporal --tsteps ${CMAKE_MATCH_1})
    else()
      set(kernel_args --kernel ${variant})
    endif()
    foreach(precision double float)
      # 500 steps on 37 x 37: the heat reaches the whole grid, so a wrong value anywhere shows up.
      heat_add_test(cuda.${variant}.${precision}.whole_grid TARGET heat_cuda LABELS cuda
        ARGS 500 37 ${kernel_args} --precision ${precision} --verify --no-csv)
      # 70 x 70: several blocks in each direction, the last ones only partly inside the grid.
      heat_add_test(cuda.${variant}.${precision}.partial_blocks TARGET heat_cuda LABELS cuda
        ARGS 41 70 ${kernel_args} --precision ${precision} --verify --no-csv)
    endforeach()
    if(HEAT_GPU_BACKEND STREQUAL "cuda")  # too slow for the emulator
      heat_add_test(cuda.${variant}.average TARGET heat_cuda EXPECT 6.01949 LABELS cuda gpu
        ARGS 1000 256 ${kernel_args} --no-csv)
      foreach(precision double float)
        heat_add_test(cuda.${variant}.${precision}.large TARGET heat_cuda LABELS cuda gpu
          ARGS 5000 150 ${kernel_args} --precision ${precision} --verify --no-csv)
      endforeach()
    endif()
  endforeach()

  heat_add_test(mpi_cuda.help TARGET heat_mpi_cuda NP 1 LABELS cuda mpi ARGS --help)
  set_tests_properties(mpi_cuda.help PROPERTIES PASS_REGULAR_EXPRESSION "Usage: heat_mpi_cuda K N")
  foreach(np 1 2 3)
    heat_add_test(mpi_cuda.np${np}.overlap TARGET heat_mpi_cuda NP ${np} LABELS cuda mpi
      ARGS 400 40 --verify --no-csv)
    heat_add_test(mpi_cuda.np${np}.blocking TARGET heat_mpi_cuda NP ${np} LABELS cuda mpi
      ARGS 400 40 --verify --no-csv --no-overlap)
  endforeach()
  heat_add_test(mpi_cuda.np4.uneven TARGET heat_mpi_cuda NP 4 LABELS cuda mpi ARGS 300 30 --verify --no-csv)
  heat_add_test(mpi_cuda.np6.one_row_strips TARGET heat_mpi_cuda NP 6 LABELS cuda mpi ARGS 50 6 --verify --no-csv)
  if(HEAT_GPU_BACKEND STREQUAL "emulator")
    # On the emulator "GPU memory" is host memory, so the CUDA-aware path can be tested with any MPI.
    heat_add_test(mpi_cuda.np2.cuda_aware TARGET heat_mpi_cuda NP 2 LABELS cuda mpi
      ARGS 400 40 --verify --no-csv --cuda-aware)
  else()
    foreach(np 1 2 4)
      heat_add_test(mpi_cuda.np${np}.average TARGET heat_mpi_cuda NP ${np} EXPECT 6.01949 LABELS cuda gpu mpi
        ARGS 1000 256 --verify --no-csv)
    endforeach()
  endif()
endif()
