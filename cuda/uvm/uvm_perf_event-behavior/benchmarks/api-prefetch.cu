// api-prefetch
//
// cudaMemPrefetchAsync in a chosen direction.  The range is first made resident
// on the source processor so the prefetch is a real migration; prefetch-driven
// migrations show up with an API cause rather than a fault cause.
//
//   usage: api-prefetch [dir=h2d|d2h] [size_mb=64] [dev=0]
#include "common.cuh"

int main(int argc, char **argv) {
    bench_dir_t dir   = arg_dir(argc, argv, 1);
    size_t      bytes = arg_bytes(argc, argv, 2, 64);
    int         dev   = (int)arg_u64(argc, argv, 3, 0);
    CUDA_CHECK(cudaSetDevice(dev));
    printf("[api-prefetch] device=%d size=%zu MiB dir=%s\n", dev, bytes / MB,
           dir_name(dir));

    char *p = nullptr;
    CUDA_CHECK(cudaMallocManaged(&p, bytes));
    make_resident_on_source(p, bytes, dir);       // resident on the source side

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));

    cudaMemLocation dst = (dir == DIR_H2D) ? loc_device(dev) : loc_host();
    printf("  [%s] prefetch %zu MiB %s\n", dir_name(dir), bytes / MB,
           dir == DIR_H2D ? "host->GPU" : "GPU->host");
    CUDA_CHECK(cudaMemPrefetchAsync(p, bytes, dst, 0, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(p));

    printf("[api-prefetch] done\n");
    return 0;
}
