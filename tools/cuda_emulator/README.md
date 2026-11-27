# CUDA emulator

A small, header-only CPU emulation of the parts of CUDA this project uses. With it the CUDA programs build and
run on machines without an NVIDIA GPU, such as a Mac or a CI runner, so the kernels' logic is tested on every
commit: indexing, boundaries, tiles, barriers and the reduction. It says nothing about performance.

```bash
cmake --preset emulator
cmake --build --preset emulator
ctest --preset emulator -L cuda                   # the CUDA tests
build/emulator/heat_cuda 300 64 --kernel temporal --verify
```

## How it works

- **Kernel launches.** [`convert_launches.cmake`](convert_launches.cmake) rewrites `kernel<<<grid, block, shared,
  stream>>>(args)` into `emu::launch(emu::cfg(grid, block, shared, stream), kernel, args)` at build time, so
  `src/cuda/kernels.cu` compiles as ordinary C++ without being edited. A `#line` directive keeps compiler errors
  pointing at the `.cu` file.
- **The CUDA API.** [`include/cuda_runtime.h`](include/cuda_runtime.h) stands in for the real header: `__global__`
  and friends become nothing, `__shared__` becomes `static`, `threadIdx`/`blockIdx`/`blockDim`/`gridDim` are
  thread-local variables, and the runtime functions (`cudaMalloc`, `cudaMemcpyAsync`, streams, events) work on
  host memory. New allocations are filled with NaN bytes, so a kernel that reads memory nobody wrote fails
  `--verify`. Streams and copies are synchronous.
- **Execution.** Blocks run one at a time, which is what makes `__shared__ static` correct. The first launch of a
  kernel runs on a pool with one `std::thread` per CUDA thread, with a barrier for `__syncthreads` and one per warp
  for `__shfl_down_sync`. If the kernel used neither, its later launches run every CUDA thread one after the other on
  the calling thread, which is much faster and gives the same result, because the threads never wait for each other.
- **Block order.** A GPU runs blocks in no particular order, so the emulator shuffles them (with a fixed seed, so
  runs are reproducible). A kernel whose result depends on the order of its blocks, such as one that writes into a
  neighbouring block's region, gives a wrong result here instead of an occasional one on a GPU.
  `CUDA_EMU_ORDER=forward|reverse|shuffle` picks the order.

## What it catches

These bugs were planted in `kernels.cu` one at a time to check that the tests notice them. Each one made
several CUDA tests fail, and the unmodified code passes:

| planted bug | caught because |
|---|---|
| temporal kernel writes one stale row too many | shuffled block order: the stale row overwrites the neighbour's correct one |
| temporal kernel writes one stale column too many | the same, along rows |
| temporal kernel: no `__syncthreads` between steps | threads read shared memory before it is written |
| shared kernel: top halo not loaded for the first row of blocks | a neighbour value is missing |
| naive kernel skips the last interior column | a column is never updated |
| reduction: second stage reads only one warp's sum | `--verify` also checks the average |
| reduction: warp shuffle skips one level | the same |

With the `sanitize` preset, AddressSanitizer and UndefinedBehaviorSanitizer also run on the emulated kernels, so an
out-of-bounds access inside a kernel stops at the exact line of `kernels.cu`.

## What it does not do

- Measure anything: timings and "bandwidth" from the emulator are meaningless.
- Run blocks or streams concurrently, so it cannot show races that need two blocks running at the same time
  (atomics, inter-block communication) or two streams racing. Ordering between streams is checked by reasoning and
  on a real GPU.
- Model warps executing in lockstep, or the GPU memory model.
- Cover CUDA beyond the subset this project uses.
