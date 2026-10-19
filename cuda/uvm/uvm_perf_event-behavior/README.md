# uvm_perf_event behavior

* Explore the behavior of uvm_perf_event_t in nvidia linux kernel module

* Trace `uvm_perf_event_notify()` and get `uvm_perf_event_data_t` data using ebpf

* The benchmarks include:

    * `malloc-managed`: malloc a managed memory

    * `first-touch-page-fault`: trigger first touch page faults in managed memory

    * `page-fault`: trigger page faults that need memory migration

    * `eviction`: exhaust device memory to trigger oom

    * `api-prefetch`: call prefetch api 

    * `api-discard`: call discard api

    * `api-discard-and-prefetch`: call discard and prefetch api

## Files

* `uvm_perf_event_notify.bt` — bpftrace probe that decodes the `uvm_perf_event_data_t`
  union for every `uvm_perf_event_notify()` call (offsets for nvidia-uvm 580.76.05).

* `benchmarks/` — one independent `.cu` per behavior above, each building into
  its own standalone binary (built against CUDA 13). Shared helpers live in
  `benchmarks/common.cuh`.

## Build

```sh
make -C benchmarks          # builds benchmarks/<name> for every <name>.cu
```

## Run

Each benchmark is a standalone program; configure it with positional arguments.

```sh
benchmarks/malloc-managed            [size_mb=64] [dev=0]
benchmarks/first-touch-page-fault    [size_mb=64] [dev=0]
benchmarks/page-fault                [size_mb=64] [iters=4] [dev=0]
benchmarks/eviction                  [dev=0]                  # size = 1.25x free mem
benchmarks/api-prefetch              [dir=h2d|d2h] [size_mb=64] [dev=0]
benchmarks/api-discard               [dir=h2d|d2h] [size_mb=64] [dev=0]
benchmarks/api-discard-and-prefetch  [dir=h2d|d2h] [size_mb=64] [dev=0]
```

For the `api-*` benchmarks, `dir` selects the migration direction; the range is
made resident on the source processor before the API call (`h2d` → host-resident
source, `d2h` → GPU-resident source).

To observe the perf events, attach the probe (needs root) in one shell and run a
benchmark in another:

```sh
sudo bpftrace uvm_perf_event_notify.bt            # shell 1
benchmarks/page-fault 16 2                         # shell 2
benchmarks/api-prefetch d2h 16                     # shell 2
```

## Expected perf events

| benchmark                  | dominant `uvm_perf_event`s                          |
|----------------------------|-----------------------------------------------------|
| `malloc-managed`           | `BLOCK_DESTROY` / `RANGE_DESTROY` at free time      |
| `first-touch-page-fault`   | `FAULT` (pages populated on the GPU, no migration)  |
| `page-fault`               | `FAULT` + `MIGRATION` both directions (fault cause) |
| `eviction`                 | `MIGRATION` with `cause=EVICTION`                   |
| `api-prefetch`             | `MIGRATION` with an API cause (`API_MIGRATE`)       |
| `api-discard`              | discard path (drops GPU-resident pages)             |
| `api-discard-and-prefetch` | discard + `MIGRATION` to the prefetch target        |