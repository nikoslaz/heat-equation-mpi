#!/usr/bin/env bash
# Times the CPU program (heat) for several process counts, with and without overlapping the halo exchange,
# prints a table and writes results/cpu-<machine>.csv. Each time is the best of REPEATS runs.
# Usage: scripts/bench_mpi.sh [build directory]           (default: build/release)
#        N=2048 K=300 PROCS="1 2 4" REPEATS=5 scripts/bench_mpi.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BUILD=${1:-build/release}
N=${N:-4096}
K=${K:-200}
PROCS=${PROCS:-"1 2 4 8"}
REPEATS=${REPEATS:-3}

cores=$(sysctl -n hw.ncpu 2> /dev/null || nproc)
machine=$(sysctl -n machdep.cpu.brand_string 2> /dev/null ||
    sed -n 's/^model name[[:space:]]*: //p' /proc/cpuinfo | head -1)
slug=$(printf '%s' "$machine" | tr '[:upper:]' '[:lower:]' | tr -cs 'a-z0-9' '-' | sed 's/^-//; s/-$//')
out="results/cpu-$slug.csv"
oversubscribe=""
if mpirun --version 2>&1 | grep -q "Open MPI"; then oversubscribe="--oversubscribe"; fi

value() { awk -v key="$1" '$1 == key { print $3 }'; }  # the number after "<key> =" in the program output

echo "machine,program,processes,overlap,N,K,seconds,glups" > "$out"
printf "Machine: %s (%s cores), N = %s, K = %s, best of %s runs\n\n" "$machine" "$cores" "$N" "$K" "$REPEATS"
printf "| processes | overlap | time (s) | GLUP/s | speedup |\n|---|---|---|---|---|\n"
for overlap in on off; do
    flag=""
    if [ "$overlap" = off ]; then flag="--no-overlap"; fi
    base=""
    for np in $PROCS; do
        best=""
        for _ in $(seq "$REPEATS"); do
            t=$(mpirun $oversubscribe -np "$np" "$BUILD/heat" "$K" "$N" --no-csv $flag | value Time)
            best=$(awk -v a="$best" -v b="$t" 'BEGIN { print (a == "" || b + 0 < a + 0) ? b : a }')
        done
        if [ -z "$base" ]; then base=$best; fi
        glups=$(awk -v n="$N" -v k="$K" -v t="$best" 'BEGIN { printf "%.4g", n * n * k / t / 1e9 }')
        speedup=$(awk -v b="$base" -v t="$best" 'BEGIN { printf "%.2f", b / t }')
        echo "\"$machine\",heat,$np,$overlap,$N,$K,$best,$glups" >> "$out"
        echo "| $np | $overlap | $best | $glups | $speedup |"
    done
done
echo
echo "Saved $out"
