// page-fault
//
// Page faults that need migration: populate on the CPU first, then read/write on
// the GPU (migrate H2D), then back on the CPU (migrate D2H).  Repeat to generate
// a stream of FAULT + MIGRATION events in both directions.
//
//   usage: page-fault [size_mb=64] [iters=4] [dev=0]
#include "common.cuh"

int main(int argc, char **argv) {
    size_t bytes = arg_bytes(argc, argv, 1, 64);
    int    iters = (int)arg_u64(argc, argv, 2, 4);
    int    dev   = (int)arg_u64(argc, argv, 3, 0);
    CUDA_CHECK(cudaSetDevice(dev));
    printf("[page-fault] device=%d size=%zu MiB iters=%d\n", dev, bytes / MB, iters);

    char *p = nullptr;
    CUDA_CHECK(cudaMallocManaged(&p, bytes));
    printf("  ping-pong %zu MiB CPU<->GPU x%d\n", bytes / MB, iters);
    for (int i = 0; i < iters; ++i) {
        cpu_touch(p, bytes);   // resident on CPU
        gpu_touch(p, bytes);   // fault + migrate H2D
        cpu_touch(p, bytes);   // fault + migrate D2H
    }
    CUDA_CHECK(cudaFree(p));

    printf("[page-fault] done\n");
    return 0;
}
