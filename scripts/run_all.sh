#!/usr/bin/env bash
# Build, check correctness, and benchmark the MPI (and, if built, CUDA) heat solver.
# Usage: ./scripts/run_all.sh            (defaults)
#        BENCH_N=2048 BENCH_K=300 ./scripts/run_all.sh
set -e
cd "$(dirname "$0")/.."

BENCH_N=${BENCH_N:-4096}
BENCH_K=${BENCH_K:-200}
CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc)

echo "== Build =="
cmake -S . -B build > /dev/null
cmake --build build

run() {  # run <np> <K> <N>
    local np=$1 extra=""
    if [ "$np" -gt "$CORES" ]; then extra="--oversubscribe"; fi
    mpirun $extra -np "$np" ./build/heat "$2" "$3"
}

echo
echo "== Correctness: N=256, K=1000 (expected Average = 6.01949) =="
for np in 1 2 4; do
    avg=$(run $np 1000 256 | grep Average | awk '{print $3}')
    if [ "$avg" = "6.01949" ]; then status="OK"; else status="MISMATCH"; fi
    echo "np=$np  Average=$avg  $status"
done

echo
echo "== Benchmark: N=$BENCH_N, K=$BENCH_K (cores on this machine: $CORES) =="
printf "%-6s %-12s %-10s %-10s\n" "np" "time (s)" "speedup" "efficiency"
base=""
{
echo "| processes | time (s) | speedup | efficiency |"
echo "|---|---|---|---|"
} > results/results.md
for np in 1 2 4 8; do
    t=$(run $np "$BENCH_K" "$BENCH_N" | grep Time | awk '{print $3}')
    if [ -z "$base" ]; then base=$t; fi
    sp=$(awk -v b="$base" -v t="$t" 'BEGIN{printf "%.2f", b/t}')
    ef=$(awk -v s="$sp" -v n="$np" 'BEGIN{printf "%.0f%%", 100*s/n}')
    printf "%-6s %-12s %-10s %-10s\n" "$np" "$t" "$sp" "$ef"
    echo "| $np | $t | $sp | $ef |" >> results/results.md
done

echo
echo "Table saved to results/results.md"

if [ -x ./build/heat_cuda ]; then
    echo
    ./scripts/test_cuda.sh
    echo
    ./scripts/bench_cuda.sh
else
    echo
    echo "(no ./build/heat_cuda: CUDA not available, skipping the GPU checks and benchmarks)"
fi
