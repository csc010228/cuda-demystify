// api-discard
//
// cudaMemDiscardBatchAsync: make the range resident on the source processor,
// then tell the driver its contents are no longer useful.  A discard lets the
// driver drop the (resident) pages without migrating their now-meaningless
// contents anywhere.
//
//   usage: api-discard [dir=h2d|d2h] [size_mb=64] [dev=0]
//          dir selects which processor's resident copy is discarded
//          (h2d -> host-resident, d2h -> GPU-resident).
#include "common.cuh"

int main(int argc, char **argv) {
    bench_dir_t dir   = arg_dir(argc, argv, 1);
    size_t      bytes = arg_bytes(argc, argv, 2, 64);
    int         dev   = (int)arg_u64(argc, argv, 3, 0);
    CUDA_CHECK(cudaSetDevice(dev));
    printf("[api-discard] device=%d size=%zu MiB dir=%s\n", dev, bytes / MB,
           dir_name(dir));

    char *p = nullptr;
    CUDA_CHECK(cudaMallocManaged(&p, bytes));
    make_resident_on_source(p, bytes, dir);       // resident on the source side

    cudaStream_t s;
    CUDA_CHECK(cudaStreamCreate(&s));             // batch APIs reject the NULL stream

    void  *dptrs[1] = {p};
    size_t sizes[1] = {bytes};
    printf("  [%s] discard %zu MiB (resident on %s)\n", dir_name(dir), bytes / MB,
           dir == DIR_H2D ? "host" : "GPU");
    CUDA_CHECK(cudaMemDiscardBatchAsync(dptrs, sizes, 1, 0, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    CUDA_CHECK(cudaStreamDestroy(s));
    CUDA_CHECK(cudaFree(p));

    printf("[api-discard] done\n");
    return 0;
}
