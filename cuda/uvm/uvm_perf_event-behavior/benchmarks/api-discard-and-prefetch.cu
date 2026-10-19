// api-discard-and-prefetch
//
// cudaMemDiscardAndPrefetchBatchAsync: make the range resident on the source
// processor, then in one fused operation discard those contents and prefetch the
// (fresh) range to the destination processor.
//
//   usage: api-discard-and-prefetch [dir=h2d|d2h] [size_mb=64] [dev=0]
#include "common.cuh"

int main(int argc, char **argv) {
    bench_dir_t dir   = arg_dir(argc, argv, 1);
    size_t      bytes = arg_bytes(argc, argv, 2, 64);
    int         dev   = (int)arg_u64(argc, argv, 3, 0);
    CUDA_CHECK(cudaSetDevice(dev));
    printf("[api-discard-and-prefetch] device=%d size=%zu MiB dir=%s\n", dev,
           bytes / MB, dir_name(dir));

    char *p = nullptr;
    CUDA_CHECK(cudaMallocManaged(&p, bytes));
    make_resident_on_source(p, bytes, dir);       // resident on the source side

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));

    void  *dptrs[1] = {p};
    size_t sizes[1] = {bytes};
    cudaMemLocation locs[1]    = {(dir == DIR_H2D) ? loc_device(dev) : loc_host()};
    size_t          locIdxs[1] = {0};

    printf("  [%s] discard+prefetch %zu MiB %s\n", dir_name(dir), bytes / MB,
           dir == DIR_H2D ? "host->GPU" : "GPU->host");
    CUDA_CHECK(cudaMemDiscardAndPrefetchBatchAsync(dptrs, sizes, 1, locs, locIdxs,
                                                   1, 0, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(p));

    printf("[api-discard-and-prefetch] done\n");
    return 0;
}
