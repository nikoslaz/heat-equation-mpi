# Benchmark results

Raw measurements, one CSV file per machine. `scripts/make_charts.py` turns them into the charts in
[`docs/images`](../docs/images), and [`docs/performance.md`](../docs/performance.md) discusses them.

| file | machine | written by |
|---|---|---|
| `cpu-apple-m4.csv` | Apple M4 (4 performance + 6 efficiency cores), macOS 15, Open MPI 5.0 | `scripts/bench_mpi.sh` |
| `gpu-tesla-t4.csv` | NVIDIA Tesla T4 on Google Colab, CUDA 13.0 | `scripts/bench_cuda.sh` |

The T4 file was written by [the Colab notebook](../colab/heat_cuda.ipynb): every kernel and precision for three
grid sizes, the temporal kernel with 1 to 8 steps per launch, and `heat_mpi_cuda` with 1, 2 and 4 processes sharing
the GPU.

## Columns

- `seconds`: duration of the time-step loop (the slowest process for MPI runs), best of several runs for the CPU.
- `glups`: billions of grid-point updates per second, `N * N * K / seconds / 1e9`.
- `effective_gbs` (GPU): the memory traffic a kernel that reads and writes the grid once per time step would need
  for that rate, `glups * 2 * bytes per value`. The temporal kernel can exceed the GPU's peak (`peak_gbs`) because
  it moves the grid once per several steps.
- `overlap`: whether the halo exchange ran overlapped with the computation of the inner rows.
