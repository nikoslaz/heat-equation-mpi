# Contributing

Issues and pull requests are welcome. [docs/development.md](docs/development.md) explains how to build, test and
benchmark the project; the short version:

```bash
cmake --preset emulator            # CPU program, plus the CUDA programs on the CPU emulator
cmake --build --preset emulator
ctest --preset emulator            # what CI runs (plus nvcc builds and sanitizers)
```

Before opening a pull request:

- Run the tests. Changes to the CUDA kernels need the `cuda` label to pass, on the emulator at least
  (`ctest --preset emulator -L cuda`), and on a GPU if you have one: [the Colab notebook](colab/heat_cuda.ipynb)
  builds any branch on a free one.
- Format the code: `clang-format -i` on the files you changed (clang-format 19, `pip install clang-format==19.1.7`).
- Run `clang-tidy -p build/emulator <files>` on the C++ files you changed.
- If you change performance, include before/after numbers from `scripts/bench_mpi.sh` or `scripts/bench_cuda.sh`,
  and the machine they come from.
