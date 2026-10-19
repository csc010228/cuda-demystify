// malloc-managed
//
// Allocate a managed region and free it.  No touch, so the interesting events
// are the *_DESTROY events emitted at free time.
//
//   usage: malloc-managed [size_mb=64] [dev=0]
#include "common.cuh"

int main(int argc, char **argv) {
    size_t bytes = arg_bytes(argc, argv, 1, 64);
    int    dev   = (int)arg_u64(argc, argv, 2, 0);
    CUDA_CHECK(cudaSetDevice(dev));
    printf("[malloc-managed] device=%d size=%zu MiB\n", dev, bytes / MB);

    char *p = nullptr;
    CUDA_CHECK(cudaMallocManaged(&p, bytes));
    printf("  allocated %zu MiB managed at %p\n", bytes / MB, (void *)p);
    CUDA_CHECK(cudaFree(p));
    printf("  freed\n");

    printf("[malloc-managed] done\n");
    return 0;
}
