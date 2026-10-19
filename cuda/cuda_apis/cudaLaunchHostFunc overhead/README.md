# `cudaLaunchHostFunc` overhead benchmark

This benchmark measures both sides of [`cudaLaunchHostFunc`](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html):

- **Host-side API overhead:** wall-clock time spent inside each `cudaLaunchHostFunc` call, plus the paired difference in total submission time with and without callbacks.
- **GPU-visible stream overhead:** paired CUDA-event time for the same kernel sequence with and without callbacks. The difference is the time for which callbacks prevent later work in that stream from running; it is reported per callback.

The callback does not invoke CUDA APIs. CUDA forbids CUDA API calls from a stream host function.

## Build

```bash
make
```

## Quick start

Run 100 kernels, put a no-op host function after each kernel, and give each kernel a nominal computational intensity of 32 FLOP/byte:

```bash
./cuda_launch_host_func_overhead \
  --kernels 100 \
  --kernel-ci 32 \
  --trials 30
```

Set the computational intensity of every kernel independently. The list length sets the kernel count:

```bash
./cuda_launch_host_func_overhead \
  --kernel-ci-list 0.25,1,4,16,64,256 \
  --callback-every 1
```

Test under controlled CPU and competing-GPU load:

```bash
./cuda_launch_host_func_overhead \
  --kernels 100 \
  --kernel-ci 32 \
  --cpu-load-threads 8 \
  --gpu-load-streams 2 \
  --gpu-load-kernels 200 \
  --gpu-load-ci 64
```

Add deliberate callback work to distinguish fixed dispatch overhead from callback execution time:

```bash
./cuda_launch_host_func_overhead --kernels 50 --callback-work-us 10
```

Use `--csv` for automation. Run `./cuda_launch_host_func_overhead --help` for every option.

## Workload model

Each thread loads one `float`, executes a dependent FMA loop, and stores one `float`. Counting one FMA as two FLOPs and the load/store as eight bytes, the requested nominal intensity maps to:

```text
FMA iterations = ceil(4 * FLOP_per_byte)
```

This is a controlled synthetic intensity, not a claim about achieved hardware FLOP/byte. Compiler output, caches, occupancy, clocks, and contention still affect elapsed time.

## Reading the output

- `cudaLaunchHostFunc API` is the most direct CPU/API measurement. It includes the small `steady_clock` measurement cost, whose median floor is printed separately.
- `Incremental submit/callback` subtracts the baseline kernel-only submission time. It can include driver queue backpressure under heavy load.
- `Incremental stall/callback` is `(callback stream time - baseline stream time) / callbacks`, using paired trials and alternating their order. It includes host-function dispatch, scheduling delay, optional callback work, and stream resumption. That is the observable cost to dependent GPU work.
- `API-return to callback-start` is queue latency, not fixed API overhead. It includes all earlier work ahead of that callback in the stream.

For stable results, use persistence mode/fixed clocks when available, avoid display GPUs, run more trials, and report the full command line, GPU/driver/runtime versions, CPU affinity, and system load. Small negative paired differences are possible when noise exceeds the overhead; increase `--trials` and kernel count in that case.
