// first-touch-page-fault
//
// The pages are never resident anywhere, so the very first GPU access populates
// them on the device.  Expect FAULT events with no preceding residency (i.e. no
// MIGRATION between two valid processors).
//
//   usage: first-touch-page-fault [size_mb=64] [dev=0]
#include "common.cuh"

int main(int argc, char **argv) {
    size_t bytes = arg_bytes(argc, argv, 1, 64);
    int    dev   = (int)arg_u64(argc, argv, 2, 0);
    CUDA_CHECK(cudaSetDevice(dev));
    printf("[first-touch-page-fault] device=%d size=%zu MiB\n", dev, bytes / MB);

    char *p = nullptr;
    CUDA_CHECK(cudaMallocManaged(&p, bytes));
    printf("  first-touch %zu MiB on GPU\n", bytes / MB);
    gpu_touch(p, bytes);
    CUDA_CHECK(cudaFree(p));

    printf("[first-touch-page-fault] done\n");
    return 0;
}
