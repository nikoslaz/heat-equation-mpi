# Parallel 2D Heat Equation (C++ / MPI / CUDA)

A solver for the 2D heat equation on a square grid, written in C++17, in three versions: distributed-memory
MPI, a single GPU with CUDA, and MPI + CUDA across several GPUs.
The top edge is held at 100, the other three edges at 0, and the interior evolves with an explicit
5-point stencil:

```
u_new[i][j] = u[i][j] + alpha * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - 4*u[i][j])
```

with `alpha = 0.2` (the explicit scheme is stable for `alpha <= 0.25`).

![Temperature after 1000 steps on a 256x256 grid](results/heat.png)

## How it works

- **Row decomposition.** The `N x N` grid is split into horizontal strips of `N / P` rows, one per MPI process.
- **Ghost rows.** Each process stores its strip in a flat `std::vector<double>` of `(N/P + 2) * N` values.
  The two extra rows hold copies of the neighbours' edge rows.
- **Halo exchange.** Before every time step, neighbouring processes swap edge rows with two
  `MPI_Sendrecv` calls. The outermost processes talk to `MPI_PROC_NULL`, so no special cases are needed.
- **Double buffering.** Two grids (`u`, `uNew`) are swapped after every step.
- **Results.** The average temperature is computed with `MPI_Reduce` (sum), the run time is the slowest
  process (`MPI_Reduce` with `MPI_MAX`, measured with `MPI_Wtime` after a barrier), and for small grids
  (`N <= 1024`) the full grid is collected with `MPI_Gather` and written to `grid.csv` in the directory the program is run from. Move it to `results/` for `scripts/plot.py`.

### CUDA versions

| file | what it is |
|---|---|
| `heat_gpu.cu`, `heat_gpu.hpp` | the CUDA kernels, behind a small C++ interface |
| `heat_cuda.cpp` | single-GPU program (`heat_cuda`) |
| `heat_mpi_cuda.cpp` | MPI + CUDA program (`heat_mpi_cuda`), one GPU per rank |
| `heat_reference.hpp` | serial CPU version, used to check the GPU results point by point |

The kernels are in their own `.cu` file and the programs are plain C++, so the MPI code never goes through `nvcc`.

**Single GPU (`heat_cuda`).** The whole grid lives in GPU memory as two buffers that swap after each step.
The grid is copied to the GPU once and back once, and the time-step loop runs entirely on the GPU
(timed with CUDA events, after one untimed warm-up step). There are three kernels (`--kernel`):

- **naive**: one thread per grid point in `32 x 8` blocks; every value is loaded from global memory, and the
  L1/L2 caches serve most of the repeated loads of neighbouring values.
- **shared**: each block first copies its tile plus a one-point halo into shared memory, then reads the
  neighbours from there. The point is to measure whether doing the caches' job by hand pays off.
- **temporal** (temporal blocking): the stencil is limited by memory bandwidth, so the way to go faster is to
  move less data. Each block loads a `32 x 64` tile into shared memory and advances it `S` steps there
  (`--tsteps S`, 1–8) before writing back. After `s` steps only points at least `s` away from the tile edge are
  still exact, so tiles overlap by `2S` and each block writes back its inner `(32-2S) x (64-2S)` points.
  The grid crosses DRAM once per `S` steps instead of once per step, in exchange for some repeated arithmetic
  in the overlaps.

Every kernel runs in `double` or `float` (`--precision`); `float` moves half the bytes. The average is computed
on the GPU with a two-level reduction (warp shuffles, then shared memory), so only one number per block is copied
back. `--verify` reruns the problem on the CPU and reports the largest difference over the whole grid.

**Several GPUs (`heat_mpi_cuda`).** Same row decomposition as the MPI version, but each rank keeps its strip and
ghost rows on its own GPU (rank `r` on a node uses GPU `r % GPU count`). Each step:

1. the interior rows, which don't need ghost rows, start computing on one CUDA stream;
2. meanwhile, on a second stream, the two edge rows are copied to pinned host memory, exchanged with
   `MPI_Sendrecv`, and copied back into the ghost rows;
3. the two edge rows are then updated on the second stream.

So the halo exchange is hidden behind the interior computation. `--no-overlap` does the steps one after another
for comparison. `--cuda-aware` passes GPU pointers straight to MPI, without the host copies (needs an MPI built
with CUDA support).

## Project layout

- `src/mpi/`: the MPI version (`main.cpp`, `heat_mpi.cpp`, `heat_mpi.hpp`)
- `src/cuda/`: the CUDA versions (see below)
- `scripts/`: `run_all.sh`, `test_cuda.sh`, `bench_cuda.sh` and `plot.py`
- `results/`: the generated tables and plot (`results.md`, `results_cuda.md`, `heat.png`, `grid.csv`)
- `colab/`: the Google Colab notebook and the script that generates it

## Build and run

Requirements: a C++17 compiler, CMake >= 3.16 and an MPI implementation (e.g. `brew install open-mpi`).

```bash
cmake -S . -B build
cmake --build build

# usage: heat K N      (K = time steps, N = grid size)
mpirun -np 4 ./build/heat 1000 256

# or via the CMake target (defaults: NP=4, K=1000, N=256)
cmake --build build --target run
```

`N` must be divisible by the number of processes.

**CUDA.** If CMake (>= 3.18) finds a CUDA compiler (`nvcc`, from the CUDA Toolkit), it also builds
`build/heat_cuda` and `build/heat_mpi_cuda`. Otherwise they are skipped and only the MPI version is built
(e.g. on macOS, which has no CUDA support).

```bash
# usage: heat_cuda K N [--kernel naive|shared|temporal] [--precision double|float] [--tsteps S] [--verify] [--no-csv]
./build/heat_cuda 1000 256
./build/heat_cuda 1000 256 --kernel temporal --tsteps 4 --verify

# usage: heat_mpi_cuda K N [--no-overlap] [--cuda-aware]
mpirun -np 2 ./build/heat_mpi_cuda 1000 256

# CUDA checks (every kernel and precision vs the CPU) and benchmarks (writes results/results_cuda.md)
./scripts/test_cuda.sh
./scripts/bench_cuda.sh

# or via the CMake targets
cmake --build build --target run_cuda
cmake --build build --target run_mpi_cuda
```

With CMake >= 3.24 the code is compiled for the GPU in the machine; to target another GPU pass
e.g. `-DCMAKE_CUDA_ARCHITECTURES=80` when configuring.

**Google Colab.** No NVIDIA GPU at hand? Upload [`colab/heat_cuda.ipynb`](colab/heat_cuda.ipynb) to
[Colab](https://colab.research.google.com) (File → Upload notebook), choose a T4 GPU runtime and run all
cells. It contains the source files, builds them, runs the checks and the benchmarks, and plots the result.
After changing the code, regenerate it with `python3 colab/make_notebook.py`.

Plot the result (needs `numpy` and `matplotlib`):

```bash
python3 scripts/plot.py
```

Everything in one go (build, correctness checks, benchmark, writes `results/results.md`; on a machine with CUDA also
runs `scripts/test_cuda.sh` and `scripts/bench_cuda.sh`):

```bash
./scripts/run_all.sh
```

## Validation

The parallel version was checked against a single-process reference implementation:
for `N = 256`, `K = 1000` the average temperature is **6.01949**, and the program prints the same value
with 1, 2 and 4 processes.

`scripts/test_cuda.sh` checks the CUDA versions:

- `heat_cuda` with every kernel gives 6.01949 for `N = 256`, `K = 1000`;
- every kernel, in double and float, matches the CPU at every grid point (`--verify`; tolerance `1e-9` for double,
  `1e-2` for float). The grid sizes are not multiples of the tile sizes, and the step counts are not multiples of
  `S`. One problem (`K = 5000`, `N = 150`) runs long enough for the heat to reach the whole grid; with few steps
  most of the grid is still exactly 0, and a wrong value there would go unnoticed;
- `heat_mpi_cuda` gives 6.01949 with 1, 2 and 4 ranks, with and without overlap.

## Performance

Strong scaling for `N = 4096`, `K = 200` (time of the slowest process):

| processes | time (s) | speedup | efficiency |
|---|---|---|---|
| 1 | 1.367 | 1.00 | 100% |
| 2 | 0.888 | 1.54 | 77% |
| 4 | 0.740 | 1.85 | 46% |
| 8 | 0.961 | 1.42 | 18% |

Speedup improves up to 4 processes and then gets worse at 8. The stencil does very little arithmetic per
byte loaded, so it is limited by memory bandwidth rather than compute, which caps the speedup well below
the process count. At 8 processes the run is likely also affected by more processes than physical cores
on the machine, and by halo-exchange overhead. Larger grids or a machine with more cores and memory
bandwidth would be needed to see better scaling.

### GPU (CUDA)

Measured on a Tesla T4 (Google Colab, CUDA 13.0, compiled with `nvcc -O3 -arch=native`). The CUDA version
gives the same `N = 256`, `K = 1000` average (**6.01949**). Time is the time-step loop only (CUDA events),
without the copies to and from the GPU.

| N | K | time (s) | effective bandwidth |
|---|---|---|---|
| 4096 | 200 | 0.240 | ~224 GB/s |
| 8192 | 200 | 0.894 | ~240 GB/s |
| 16384 | 100 | 1.773 | ~242 GB/s |

Effective bandwidth counts one read and one write of 8 bytes per grid point per step
(`2 * N^2 * 8 * K / time`). It assumes neighbouring values come from cache.

For `N = 4096`, `K = 200` the GPU takes 0.240 s, against 1.367 s for 1 MPI process (**5.7x** faster) and 0.740 s
for the best MPI run with 4 processes (**3.1x** faster). The T4's peak memory bandwidth is about 320 GB/s, so
the simple kernel already reaches roughly 70–75% of it. That fits the stencil being limited by memory
bandwidth: a kernel that reads and writes the grid once per step can get at most about another 1.3x. Going
further means moving fewer bytes per step, which is what the `float` and temporal-blocking versions do;
`bench_cuda.sh` compares all of them. The CPU and GPU runs were on different machines (a Mac for MPI, Colab
for CUDA), so the comparison is indicative.

## Possible next steps

- Handle `N` not divisible by the number of processes.
- Overlap communication and computation (non-blocking `MPI_Isend`/`MPI_Irecv` on the interior rows).
- 2D domain decomposition to reduce the amount of data exchanged per process.
- Add OpenMP inside each process (hybrid MPI + OpenMP).
- Profile the CUDA kernels with Nsight Compute (the kernels are built with `-lineinfo` for this).
- Use temporal blocking in the MPI + CUDA version too (needs `S` ghost rows per side, exchanged every `S` steps).
- Run `heat_mpi_cuda` on a machine with several GPUs to measure real multi-GPU scaling.
