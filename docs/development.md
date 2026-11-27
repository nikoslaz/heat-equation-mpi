# Development

## Requirements

- A C++17 compiler (GCC, Clang or Apple Clang) and CMake 3.22 or newer
- An MPI implementation: Open MPI (`brew install open-mpi`, `apt install libopenmpi-dev openmpi-bin`) or MPICH
- For the GPU programs: the CUDA Toolkit, 12 or newer, and an NVIDIA GPU. Without one, use the CPU emulator
  (below) or [the Colab notebook](../colab/heat_cuda.ipynb)
- For the figures: Python 3 with `pip install -r scripts/requirements.txt`

GoogleTest is downloaded by CMake if it is not installed.

## Building

The build configurations are CMake presets ([`CMakePresets.json`](../CMakePresets.json)); each builds into
`build/<preset>`:

| preset | what it builds |
|---|---|
| `release` | optimised; the CUDA programs too when `nvcc` is found |
| `debug` | the same, unoptimised with debug information |
| `cuda` | like `release`, but requires `nvcc` |
| `emulator` | the CUDA programs against the CPU emulator, for machines without a GPU |
| `sanitize` | debug build with AddressSanitizer and UndefinedBehaviorSanitizer, CUDA code on the emulator |
| `ci` | what CI builds: `emulator` with warnings as errors |

```bash
cmake --preset emulator
cmake --build --preset emulator -j
```

Options, all with `-D` at configure time: `HEAT_CUDA` (`AUTO`, `ON`, `OFF`, `EMULATOR`), `HEAT_WERROR`,
`HEAT_SANITIZE`, `HEAT_BUILD_TESTS`, and `CMAKE_CUDA_ARCHITECTURES` (for example `75` for a T4, `80` for an A100;
by default the GPU in the machine).

## Testing

```bash
ctest --preset emulator               # everything this build has
ctest --preset emulator -L unit       # only the unit tests
ctest --preset emulator -L cuda -j 6  # only the CUDA programs
```

| label | what it covers |
|---|---|
| `unit` | GoogleTest tests of the shared code: the command-line parser, the row decomposition, the stencil, the output ([`tests/unit`](../tests/unit)) |
| `cpu` | `heat` with 1 to 7 processes, both exchange modes, uneven strips and one-row strips |
| `cuda` | `heat_cuda` with every kernel and precision, and `heat_mpi_cuda` with 1 to 6 processes, on a GPU or the emulator |
| `gpu` | larger problems that only run on a real GPU |
| `mpi` | every test that runs under `mpiexec` |

The program tests rely on `--verify`, which every program has: it compares every grid point and the average with
a serial run and exits with 1 on a mismatch. Grid sizes are chosen so that tiles and strips do not divide them
evenly, and some problems run long enough for the heat to reach the whole grid; with few steps most of the grid is
still exactly 0 and a wrong value there would not show.

## Testing CUDA code without a GPU

[`tools/cuda_emulator`](../tools/cuda_emulator) is a small CPU emulation of the CUDA features this project uses. It
runs blocks in shuffled order, so a kernel that depends on the order of its blocks fails, and its allocations start
as NaN, so reading memory nobody wrote fails `--verify`. Seven bugs planted in the kernels to test it were all
caught. Its README explains how it works and what it cannot check.

## Sanitizers

```bash
cmake --preset sanitize && cmake --build --preset sanitize -j && ctest --preset sanitize
```

AddressSanitizer and UndefinedBehaviorSanitizer then check the CPU code, the program drivers and the emulated
kernels. Leak checking is off in the test preset, because MPI libraries keep memory until exit.

## Formatting and static analysis

```bash
pip install clang-format==19.1.7 clang-tidy==19.1.0
git ls-files '*.cpp' '*.hpp' '*.cu' '*.h' | xargs clang-format -i
git ls-files 'src/*.cpp' 'tests/*.cpp' | xargs clang-tidy -p build/emulator
```

The style is in [`.clang-format`](../.clang-format) and the checks in [`.clang-tidy`](../.clang-tidy). On macOS,
the pip build of clang-tidy needs `--extra-arg=-isysroot$(xcrun --show-sdk-path)` to find the C++ library.

## Continuous integration

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) runs on every push and pull request:

| job | what it does |
|---|---|
| Test (gcc), Test (clang) | builds with warnings as errors and runs every test, the CUDA code on the emulator |
| Sanitizers | the `sanitize` preset and its tests |
| CUDA 12.9.1 build, CUDA 13.0.2 build | compiles the CUDA code with `nvcc` for Turing, Ampere and Hopper GPUs and runs the tests that need no GPU (the runners have none) |
| Format and static analysis | `clang-format --dry-run --Werror` and `clang-tidy` |

## Benchmarks and figures

| script | what it does |
|---|---|
| [`scripts/run_all.sh`](../scripts/run_all.sh) | build, test and benchmark everything this machine supports |
| [`scripts/bench_mpi.sh`](../scripts/bench_mpi.sh) | times `heat` for several process counts, writes `results/cpu-<machine>.csv` |
| [`scripts/bench_cuda.sh`](../scripts/bench_cuda.sh) | times every GPU kernel, precision and step count, and `heat_mpi_cuda`; writes `results/gpu-<device>.csv` |
| [`scripts/make_charts.py`](../scripts/make_charts.py) | draws the charts in `docs/images` from `results/*.csv` |
| [`scripts/make_animation.py`](../scripts/make_animation.py) | the animation in the README (each frame computed by `heat`) and the social preview image |
| [`scripts/plot_grid.py`](../scripts/plot_grid.py) | draws a `grid.csv` written by any program |
| [`colab/make_notebook.py`](../colab/make_notebook.py) | regenerates the Colab notebook |

## Repository layout

```
src/core/            code shared by all programs: problem, stencil, command line, output
src/mpi/             the CPU program (heat) and the MPI helpers
src/cuda/            the CUDA kernels and the two GPU programs (heat_cuda, heat_mpi_cuda)
tests/               GoogleTest unit tests and the program tests (tests/CMakeLists.txt)
tools/cuda_emulator/ CPU emulator of CUDA, for testing without a GPU
cmake/               CMake logic for the CUDA programs and the emulator
scripts/             benchmarks and figures
results/             measured benchmark results (CSV)
docs/                documentation and figures
colab/               the Google Colab notebook
```
