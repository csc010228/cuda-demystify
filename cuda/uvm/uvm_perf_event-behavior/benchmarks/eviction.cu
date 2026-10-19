// eviction
//
// Oversubscribe device memory: allocate more managed memory than fits on the GPU
// and make all of it resident on the device.  Once the device fills, the driver
// evicts older pages back to host -> MIGRATION with cause=EVICTION.
//
// Sizes itself to 1.25x of free device memory.
//
//   usage: eviction [dev=0]
#include "common.cuh"

int main(int argc, char **argv) {
    int dev = (int)arg_u64(argc, argv, 1, 0);
    CUDA_CHECK(cudaSetDevice(dev));
    printf("[eviction] device=%d\n", dev);

    size_t freeB = 0, totalB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totalB));
    size_t bytes = (size_t)(freeB * 5 / 4);      // 1.25x free -> guaranteed oversubscribe
    bytes = (bytes / PAGE) * PAGE;
    printf("  device %d: free=%zu MiB total=%zu MiB -> oversubscribe %zu MiB\n",
           dev, freeB / MB, totalB / MB, bytes / MB);

    char *p = nullptr;
    CUDA_CHECK(cudaMallocManaged(&p, bytes));

    // Stream through the whole region in device-sized chunks twice.  The second
    // pass re-faults pages that were evicted during the first, exercising the
    // eviction path heavily.
    size_t chunk = (freeB / 2 / PAGE) * PAGE;
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t off = 0; off < bytes; off += chunk) {
            size_t n = (off + chunk <= bytes) ? chunk : (bytes - off);
            gpu_touch(p + off, n);
        }
        printf("  pass %d done\n", pass);
    }
    CUDA_CHECK(cudaFree(p));

    printf("[eviction] done\n");
    return 0;
}
