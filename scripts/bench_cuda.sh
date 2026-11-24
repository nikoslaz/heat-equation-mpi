#!/usr/bin/env bash
# Benchmark the CUDA programs and write the tables to results/results_cuda.md (needs an NVIDIA GPU).
# Usage: ./scripts/bench_cuda.sh                       (defaults)
#        SIZES="4096 8192" K=100 ./scripts/bench_cuda.sh
set -e
cd "$(dirname "$0")/.."
BIN=${BIN:-./build}
SIZES=${SIZES:-"4096 8192 16384"}
K=${K:-200}
TSTEPS=${TSTEPS:-4}       # time steps per launch for the temporal kernel in the main table
SWEEP_N=${SWEEP_N:-8192}  # grid size for the temporal-steps sweep and the MPI + CUDA runs
OUT=results/results_cuda.md
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu)

value() {  # value <key> : print the number after "<key> =" in the program output on stdin
    awk -v key="$1" '$1 == key {print $3}'
}
ratio() { awk -v a="$1" -v b="$2" 'BEGIN {printf "%.2f", a / b}'; }
fmt() { awk -v x="$1" 'BEGIN {printf "%.4g", x}'; }

gpu=$("$BIN/heat_cuda" 1 16 --no-csv | sed -n 's/^GPU: //p')
peak=$("$BIN/heat_cuda" 1 16 --no-csv | sed -n 's/.*device peak \([0-9.]*\) GB.*/\1/p')

{
    echo "GPU: $gpu (peak memory bandwidth ${peak:-?} GB/s), K = $K time steps"
    echo
    echo "### Kernels"
    echo
    echo "| N | precision | kernel | time (s) | GLUP/s | effective GB/s | speedup vs naive |"
    echo "|---|---|---|---|---|---|---|"
} > "$OUT"

echo "== Kernels (K=$K) =="
for N in $SIZES; do
    for precision in double float; do
        base=""
        for kernel in naive shared temporal; do
            out=$("$BIN/heat_cuda" "$K" "$N" --kernel "$kernel" --precision "$precision" --tsteps "$TSTEPS" --no-csv)
            t=$(echo "$out" | value Time)
            g=$(echo "$out" | value Throughput)
            bw=$(echo "$out" | value Bandwidth)
            if [ -z "$base" ]; then base=$t; fi
            label=$kernel
            if [ "$kernel" = temporal ]; then label="temporal (S=$TSTEPS)"; fi
            row="| $N | $precision | $label | $(fmt "$t") | $(fmt "$g") | $(fmt "$bw") | $(ratio "$base" "$t") |"
            echo "$row"
            echo "$row" >> "$OUT"
        done
    done
done

{
    echo
    echo "### Temporal blocking: time steps per launch (N = $SWEEP_N)"
    echo
    echo "| S | double: time (s) | double: GLUP/s | float: time (s) | float: GLUP/s |"
    echo "|---|---|---|---|---|"
} >> "$OUT"

echo
echo "== Temporal blocking sweep (N=$SWEEP_N) =="
for S in 1 2 3 4 5 6 7 8; do
    row="| $S |"
    for precision in double float; do
        out=$("$BIN/heat_cuda" "$K" "$SWEEP_N" --kernel temporal --tsteps "$S" --precision "$precision" --no-csv)
        row="$row $(fmt "$(echo "$out" | value Time)") | $(fmt "$(echo "$out" | value Throughput)") |"
    done
    echo "$row"
    echo "$row" >> "$OUT"
done

if [ -x "$BIN/heat_mpi_cuda" ] && command -v mpirun > /dev/null; then
    gpus=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    {
        echo
        echo "### MPI + CUDA (N = $SWEEP_N, double, naive kernel, $gpus GPU(s); ranks share GPUs round-robin)"
        echo
        echo "| ranks | overlap | time (s) |"
        echo "|---|---|---|"
    } >> "$OUT"
    echo
    echo "== MPI + CUDA (N=$SWEEP_N, $gpus GPU(s)) =="
    for np in 1 2 4; do
        extra=""
        if [ "$np" -gt "$CORES" ]; then extra="--oversubscribe"; fi
        for mode in on off; do
            flag=""
            if [ "$mode" = off ]; then flag="--no-overlap"; fi
            t=$(mpirun $extra -np "$np" "$BIN/heat_mpi_cuda" "$K" "$SWEEP_N" $flag | value Time)
            row="| $np | $mode | $(fmt "$t") |"
            echo "$row"
            echo "$row" >> "$OUT"
        done
    done
fi

echo
echo "Tables saved to $OUT"
