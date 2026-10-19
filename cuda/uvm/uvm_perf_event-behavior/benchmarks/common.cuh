// common.cuh
//
// Shared helpers for the uvm_perf_event-behavior benchmarks.  Each benchmark is
// its own independent .cu / binary; they only share this header.  Configuration
// is passed positionally on the command line (argc/argv) — see each main().
//
// Built against CUDA 13 (cudaMemPrefetchAsync takes a cudaMemLocation, and the
// batch discard/prefetch APIs exist).  Driver/UVM 580.76.05.
#pragma once

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "%s:%d  %s -> %s\n", __FILE__, __LINE__, #call,    \
                    cudaGetErrorString(_e));                                   \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

static const size_t PAGE = 4096;            // host/UVM CPU page
static const size_t MB   = 1024 * 1024;

// Write one byte into every 4 KB page so each page is faulted in / made
// resident on whichever processor runs the kernel.
__global__ void touch_pages(char *p, size_t bytes) {
    size_t pages  = bytes / PAGE;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < pages;
         i += stride) {
        p[i * PAGE] = (char)(i & 0xff);
    }
}

static inline void gpu_touch(char *p, size_t bytes, cudaStream_t s = 0) {
    touch_pages<<<256, 256, 0, s>>>(p, bytes);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(s));
}

static inline void cpu_touch(char *p, size_t bytes) {
    for (size_t i = 0; i < bytes; i += PAGE)
        p[i] = (char)(i & 0xff);
}

// ---- argv parsing ---------------------------------------------------------

// Positional unsigned arg argv[i] (1-based as in argv), or def if not present.
static inline size_t arg_u64(int argc, char **argv, int i, size_t def) {
    return (i < argc) ? (size_t)strtoull(argv[i], nullptr, 10) : def;
}

// Allocation size: argv[i] is a count of MiB (default def_mb), page-aligned.
static inline size_t arg_bytes(int argc, char **argv, int i, size_t def_mb) {
    size_t bytes = arg_u64(argc, argv, i, def_mb) * MB;
    return (bytes / PAGE) * PAGE;
}

// ---- migration direction --------------------------------------------------

enum bench_dir_t { DIR_H2D, DIR_D2H };

static inline const char *dir_name(bench_dir_t d) {
    return d == DIR_H2D ? "h2d" : "d2h";
}

// Parse a direction arg ("h2d" default, or "d2h"); exits on anything else.
static inline bench_dir_t arg_dir(int argc, char **argv, int i) {
    if (i >= argc) return DIR_H2D;
    if (strcmp(argv[i], "h2d") == 0 || strcmp(argv[i], "H2D") == 0) return DIR_H2D;
    if (strcmp(argv[i], "d2h") == 0 || strcmp(argv[i], "D2H") == 0) return DIR_D2H;
    fprintf(stderr, "error: direction must be 'h2d' or 'd2h', got '%s'\n", argv[i]);
    std::exit(2);
}

// Make the whole range resident on the source processor of the given direction
// (host for h2d, device for d2h) before an API call migrates it the other way.
static inline void make_resident_on_source(char *p, size_t bytes, bench_dir_t d) {
    if (d == DIR_H2D)
        cpu_touch(p, bytes);   // source = host
    else
        gpu_touch(p, bytes);   // source = device
}

// ---- locations ------------------------------------------------------------

static inline cudaMemLocation loc_device(int dev) {
    cudaMemLocation l{};
    l.type = cudaMemLocationTypeDevice;
    l.id   = dev;
    return l;
}

static inline cudaMemLocation loc_host() {
    cudaMemLocation l{};
    l.type = cudaMemLocationTypeHost;
    l.id   = 0;
    return l;
}
