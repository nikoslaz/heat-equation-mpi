#!/usr/bin/env bash
# Benchmarks the CUDA programs (needs an NVIDIA GPU), prints tables and writes results/gpu-<device>.csv:
#   1. every kernel in double and float, for several grid sizes,
#   2. the temporal kernel with 1 to 8 time steps per launch,
#   3. heat_mpi_cuda with 1, 2 and 4 processes (they share the GPUs round-robin).
# Usage: scripts/bench_cuda.sh [build directory]            (default: build/cuda)
#        SIZES="4096 8192" K=100 scripts/bench_cuda.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${1:-build/cuda}
SIZES=${SIZES:-"4096 8192 16384"}
K=${K:-200}
TSTEPS=${TSTEPS:-4}       # time steps per launch of the temporal kernel in table 1
SWEEP_N=${SWEEP_N:-8192}  # grid size for tables 2 and 3

value() { awk -v key="$1" '$1 == key { print $3 }'; }  # the number after "<key> =" in the program output
ratio() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.2f", a / b }'; }

info=$("$BUILD/heat_cuda" 1 16 --no-csv)
device=$(echo "$info" | sed -n 's/^Device *= //p')
peak=$(echo "$info" | sed -n 's/.* of the \([0-9.]*\) GB\/s peak.*/\1/p')
slug=$(printf '%s' "$device" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/^-//; s/-$//')
out="results/gpu-$slug.csv"
oversubscribe=""
if mpirun --version 2>&1 | grep -q "Open MPI"; then oversubscribe="--oversubscribe"; fi
echo "device,peak_gbs,program,kernel,tsteps,precision,processes,overlap,N,K,seconds,glups,effective_gbs" > "$out"

# One heat_cuda run, appended to the CSV; prints "seconds glups effective_gbs".
run_cuda() {  # run_cuda <N> <kernel> <tsteps> <precision>
    local n=$1 kernel=$2 s=$3 precision=$4 result t g bw
    result=$("$BUILD/heat_cuda" "$K" "$n" --kernel "$kernel" --tsteps "$s" --precision "$precision" --no-csv)
    t=$(echo "$result" | value Time)
    g=$(echo "$result" | value Rate)
    bw=$(echo "$result" | value Bandwidth)
    if [ "$kernel" != temporal ]; then s=""; fi
    echo "\"$device\",$peak,heat_cuda,$kernel,$s,$precision,1,,$n,$K,$t,$g,$bw" >> "$out"
    echo "$t $g $bw"
}

echo "Device: $device (peak memory bandwidth ${peak:-?} GB/s), K = $K"
echo
echo "### Kernels"
echo
echo "| N | precision | kernel | time (s) | GLUP/s | effective GB/s | speedup vs naive |"
echo "|---|---|---|---|---|---|---|"
for n in $SIZES; do
    for precision in double float; do
        base=""
        for kernel in naive shared temporal; do
            read -r t g bw < <(run_cuda "$n" "$kernel" "$TSTEPS" "$precision")
            if [ -z "$base" ]; then base=$t; fi
            label=$kernel
            if [ "$kernel" = temporal ]; then label="temporal (S=$TSTEPS)"; fi
            echo "| $n | $precision | $label | $t | $g | $bw | $(ratio "$base" "$t") |"
        done
    done
done

echo
echo "### Temporal kernel: time steps per launch (N = $SWEEP_N)"
echo
echo "| S | double: time (s) | double: GLUP/s | float: time (s) | float: GLUP/s |"
echo "|---|---|---|---|---|"
for s in 1 2 3 4 5 6 7 8; do
    read -r td gd _ < <(run_cuda "$SWEEP_N" temporal "$s" double)
    read -r tf gf _ < <(run_cuda "$SWEEP_N" temporal "$s" float)
    echo "| $s | $td | $gd | $tf | $gf |"
done

if [ -x "$BUILD/heat_mpi_cuda" ] && command -v mpirun > /dev/null; then
    gpus=$(nvidia-smi -L 2> /dev/null | wc -l | tr -d ' ')
    echo
    echo "### MPI + CUDA (N = $SWEEP_N, double, naive kernel, $gpus GPU(s))"
    echo
    echo "| processes | overlap | time (s) | GLUP/s |"
    echo "|---|---|---|---|"
    for np in 1 2 4; do
        for overlap in on off; do
            flag=""
            if [ "$overlap" = off ]; then flag="--no-overlap"; fi
            result=$(mpirun $oversubscribe -np "$np" "$BUILD/heat_mpi_cuda" "$K" "$SWEEP_N" --no-csv $flag)
            t=$(echo "$result" | value Time)
            g=$(echo "$result" | value Rate)
            bw=$(awk -v g="$g" 'BEGIN { printf "%.4g", g * 16 }')
            echo "\"$device\",$peak,heat_mpi_cuda,naive,,double,$np,$overlap,$SWEEP_N,$K,$t,$g,$bw" >> "$out"
            echo "| $np | $overlap | $t | $g |"
        done
    done
fi

echo
echo "Saved $out"
