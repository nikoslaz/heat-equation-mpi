# Rewrites CUDA kernel launches, kernel<<<config>>>(args), into emu::launch(emu::cfg(config), kernel, args),
# so that a .cu file compiles as ordinary C++ against the CPU emulator in include/cuda_runtime.h.
#
# Usage: cmake -DINPUT=kernels.cu -DOUTPUT=kernels.cpp -P convert_launches.cmake

if(NOT INPUT OR NOT OUTPUT)
  message(FATAL_ERROR "usage: cmake -DINPUT=<file.cu> -DOUTPUT=<file.cpp> -P convert_launches.cmake")
endif()

file(READ "${INPUT}" source)
string(REGEX REPLACE
  "([A-Za-z_][A-Za-z0-9_:]*(<[A-Za-z0-9_, ]*>)?)<<<([^>]*)>>>\\("
  "emu::launch(emu::cfg(\\3), \\1, "
  converted "${source}")
if(converted MATCHES "<<<")
  message(FATAL_ERROR "convert_launches: a kernel launch in ${INPUT} has an unexpected form")
endif()

# The #line directive makes compiler errors point at the original .cu file.
file(WRITE "${OUTPUT}"
  "// Generated from ${INPUT} by tools/cuda_emulator/convert_launches.cmake. Do not edit.\n"
  "#line 1 \"${INPUT}\"\n"
  "${converted}")
