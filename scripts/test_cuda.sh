#!/usr/bin/env bash
# Correctness checks for the CUDA programs (needs an NVIDIA GPU; build first with CMake).
# Usage: ./scripts/test_cuda.sh
cd "$(dirname "$0")/.."
BIN=${BIN:-./build}
EXPECTED=6.01949
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu)
failures=0

report() {  # report <exit status> <text>
    if [ "$1" -eq 0 ]; then echo "  OK      $2"; else echo "  FAILED  $2"; failures=$((failures + 1)); fi
}

mpi() {  # mpi <np> <command...>: run with mpirun, oversubscribing if there are fewer cores
    local np=$1 extra=""
    shift
    if [ "$np" -gt "$CORES" ]; then extra="--oversubscribe"; fi
    mpirun $extra -np "$np" "$@"
}

echo "== heat_cuda: average for N=256, K=1000 (expected $EXPECTED) =="
for kernel in naive shared temporal; do
    avg=$("$BIN/heat_cuda" 1000 256 --kernel "$kernel" --no-csv | awk '$1 == "Average" {print $3}')
    [ "$avg" = "$EXPECTED" ]
    report $? "--kernel $kernel: Average = $avg"
done

# Every grid point against a CPU run. N is not a multiple of any tile size, and K is not a
# multiple of the temporal step counts. K=5000, N=150 lets the heat reach the whole grid, so a
# mistake anywhere shows up (with few steps most of the grid is still exactly 0).
echo
echo "== heat_cuda: every grid point vs the CPU =="
for problem in "257 300" "5000 150"; do
    for precision in double float; do
        for args in "--kernel naive" "--kernel shared" "--kernel temporal --tsteps 1" \
                    "--kernel temporal --tsteps 3" "--kernel temporal --tsteps 8"; do
            line=$("$BIN/heat_cuda" $problem $args --precision $precision --verify --no-csv | grep '^Verify')
            echo "$line" | grep -q ' OK$'
            report $? "K N = $problem, $args --precision $precision: ${line#Verify: }"
        done
    done
done

if [ -x "$BIN/heat_mpi_cuda" ] && command -v mpirun > /dev/null; then
    echo
    echo "== heat_mpi_cuda: average for N=256, K=1000 (expected $EXPECTED; ranks may share one GPU) =="
    for np in 1 2 4; do
        for mode in "" "--no-overlap"; do
            avg=$(mpi "$np" "$BIN/heat_mpi_cuda" 1000 256 $mode | awk '$1 == "Average" {print $3}')
            [ "$avg" = "$EXPECTED" ]
            report $? "np=$np ${mode:-overlap}: Average = $avg"
        done
    done
fi

echo
if [ "$failures" -eq 0 ]; then
    echo "All CUDA checks passed"
else
    echo "$failures CUDA check(s) FAILED"
    exit 1
fi
