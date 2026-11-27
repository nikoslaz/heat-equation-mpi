# The CUDA programs, heat_cuda (one GPU) and heat_mpi_cuda (MPI + CUDA), are built when nvcc is available.
# With HEAT_CUDA=EMULATOR they are built against tools/cuda_emulator instead, which runs the kernels on CPU
# threads, so the CUDA code can be compiled and tested on machines without an NVIDIA GPU.
#
# Sets HEAT_GPU_BACKEND to "cuda", "emulator" or "" (no CUDA programs).

set(HEAT_GPU_BACKEND "")
if(HEAT_CUDA STREQUAL "EMULATOR")
  set(HEAT_GPU_BACKEND emulator)
elseif(NOT HEAT_CUDA STREQUAL "OFF")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    set(HEAT_GPU_BACKEND cuda)
  elseif(HEAT_CUDA STREQUAL "ON")
    message(FATAL_ERROR "HEAT_CUDA is ON, but no CUDA compiler was found")
  else()
    message(STATUS "No CUDA compiler found: building the CPU program only "
                   "(-DHEAT_CUDA=EMULATOR builds the CUDA programs against a CPU emulator)")
  endif()
endif()

if(HEAT_GPU_BACKEND STREQUAL "cuda")
  # Compile for the GPU in this machine unless an architecture is given, e.g. -DCMAKE_CUDA_ARCHITECTURES=75.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  add_library(heat_gpu STATIC src/cuda/kernels.cu)
  set_target_properties(heat_gpu PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
  # -lineinfo maps profiler results (Nsight Compute) back to source lines.
  target_compile_options(heat_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xcompiler=-Wall,-Wextra>)
  target_link_libraries(heat_gpu PUBLIC heat_core CUDA::cudart)

elseif(HEAT_GPU_BACKEND STREQUAL "emulator")
  set(emulator_dir ${PROJECT_SOURCE_DIR}/tools/cuda_emulator)
  set(converted ${PROJECT_BINARY_DIR}/cuda_emulator/kernels.cpp)
  add_custom_command(
    OUTPUT ${converted}
    COMMAND ${CMAKE_COMMAND} -DINPUT=${PROJECT_SOURCE_DIR}/src/cuda/kernels.cu -DOUTPUT=${converted}
            -P ${emulator_dir}/convert_launches.cmake
    DEPENDS ${PROJECT_SOURCE_DIR}/src/cuda/kernels.cu ${emulator_dir}/convert_launches.cmake
    COMMENT "Converting the CUDA kernel launches for the CPU emulator")

  find_package(Threads REQUIRED)
  add_library(heat_gpu STATIC ${converted})
  target_include_directories(heat_gpu PUBLIC ${emulator_dir}/include)  # its cuda_runtime.h
  target_compile_features(heat_gpu PUBLIC cxx_std_20)
  # GCC does not know CUDA's "#pragma unroll", which means nothing on the CPU anyway.
  target_compile_options(heat_gpu PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wno-unknown-pragmas>)
  target_link_libraries(heat_gpu PUBLIC heat_core Threads::Threads)
endif()

if(HEAT_GPU_BACKEND)
  add_executable(heat_cuda src/cuda/heat_cuda.cpp)
  target_link_libraries(heat_cuda PRIVATE heat_gpu)

  add_executable(heat_mpi_cuda src/cuda/heat_mpi_cuda.cpp)
  target_link_libraries(heat_mpi_cuda PRIVATE heat_gpu heat_mpi_utils)
endif()
