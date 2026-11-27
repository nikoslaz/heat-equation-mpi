#!/usr/bin/env bash
# Builds, tests and benchmarks everything this machine supports.
#   scripts/run_all.sh                   release build; the CUDA programs are included when nvcc is found
#   PRESET=emulator scripts/run_all.sh   CUDA programs on the CPU emulator (tests only, no GPU benchmark)
set -euo pipefail
cd "$(dirname "$0")/.."
PRESET=${PRESET:-release}

echo "== Build ($PRESET) =="
cmake --preset "$PRESET"
cmake --build --preset "$PRESET" -j

echo
echo "== Tests =="
ctest --preset "$PRESET" -j 4

echo
echo "== CPU benchmark =="
scripts/bench_mpi.sh "build/$PRESET"

if [ -x "build/$PRESET/heat_cuda" ] && command -v nvidia-smi > /dev/null; then
    echo
    echo "== GPU benchmark =="
    scripts/bench_cuda.sh "build/$PRESET"
fi
