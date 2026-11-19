# Parallel 2D Heat Equation (C++ / MPI / CUDA)

A solver for the 2D heat equation on a square grid, written in C++17, with a distributed-memory MPI
version and a single-GPU CUDA version.
The top edge is held at 100, the other three edges at 0, and the interior evolves with an explicit
5-point stencil:

```
u_new[i][j] = u[i][j] + alpha * (u[i-1][j] + u[i+1][j] + u[i][j-1] + u[i][j+1] - 4*u[i][j])
```

with `alpha = 0.2` (the explicit scheme is stable for `alpha <= 0.25`).

![Temperature after 1000 steps on a 256x256 grid](heat.png)

## How it works

- **Row decomposition.** The `N x N` grid is split into horizontal strips of `N / P` rows, one per MPI process.
- **Ghost rows.** Each process stores its strip in a flat `std::vector<double>` of `(N/P + 2) * N` values.
  The two extra rows hold copies of the neighbours' edge rows.
- **Halo exchange.** Before every time step, neighbouring processes swap edge rows with two
  `MPI_Sendrecv` calls. The outermost processes talk to `MPI_PROC_NULL`, so no special cases are needed.
- **Double buffering.** Two grids (`u`, `uNew`) are swapped after every step.
- **Results.** The average temperature is computed with `MPI_Reduce` (sum), the run time is the slowest
  process (`MPI_Reduce` with `MPI_MAX`, measured with `MPI_Wtime` after a barrier), and for small grids
  (`N <= 1024`) the full grid is collected with `MPI_Gather` and written to `grid.csv` for plotting.

### CUDA version

`heat_cuda.cu` runs the same scheme on one NVIDIA GPU:

- The whole `N x N` grid lives in GPU memory as two flat `double` buffers (`u`, `uNew`), swapped by pointer after every step.
- The kernel `heat_step` uses one thread per grid point in `32 x 8` thread blocks (32 threads along a row,
  so neighbouring threads read neighbouring memory). Threads on the boundary do nothing, so the fixed edge
  values are never overwritten.
- The grid is copied to the GPU once at the start and back once at the end. The time step loop runs
  entirely on the GPU and is timed with CUDA events.
- The output is the same as the MPI version (`Average`, `Time`, and `grid.csv` for `N <= 1024`).

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

**CUDA.** If CMake finds a CUDA compiler (`nvcc`, from the CUDA Toolkit), it also builds `build/heat_cuda`;
otherwise it is skipped and only the MPI version is built (e.g. on macOS, which has no CUDA support).

```bash
# usage: heat_cuda K N
./build/heat_cuda 1000 256

# or via the CMake target
cmake --build build --target run_cuda
```

With CMake >= 3.24 the code is compiled for the GPU in the machine; to target another GPU pass
e.g. `-DCMAKE_CUDA_ARCHITECTURES=80` when configuring.

Plot the result (needs `numpy` and `matplotlib`):

```bash
python3 plot.py
```

Everything in one go (build, correctness checks, benchmark, writes `results.md`):

```bash
./run_all.sh
```

## Validation

The parallel version was checked against a single-process reference implementation:
for `N = 256`, `K = 1000` the average temperature is **6.01949**, and the program prints the same value
with 1, 2 and 4 processes. `run_all.sh` also checks the CUDA version against the same value when it is built.

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
bandwidth, and it means further kernel tuning can give at most about another 1.3x. The CPU and GPU runs were
on different machines (a Mac for MPI, Colab for CUDA), so the comparison is indicative.

## Possible next steps

- Handle `N` not divisible by the number of processes.
- Overlap communication and computation (non-blocking `MPI_Isend`/`MPI_Irecv` on the interior rows).
- 2D domain decomposition to reduce the amount of data exchanged per process.
- Add OpenMP inside each process (hybrid MPI + OpenMP).
- Multi-GPU: one GPU per MPI rank, exchanging ghost rows between GPUs (CUDA-aware MPI).
- Shared-memory tiling in the CUDA kernel, and a GPU-side reduction for the average.
