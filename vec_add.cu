#include <cuda_runtime.h>
#include <iostream>

__global__ void vec_add(const float *a,
                        const float *b,
                        float *c,
                        int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    const int N = 1024;
    const size_t SIZE = N * sizeof(float);

    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    float *d_a;
    float *d_b;
    float *d_c;

    cudaMalloc(&d_a, SIZE);
    cudaMalloc(&d_b, SIZE);
    cudaMalloc(&d_c, SIZE);

    cudaMemcpy(d_a, h_a, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, SIZE, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (N + block - 1) / block;

    vec_add<<<grid, block>>>(d_a, d_b, d_c, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, SIZE, cudaMemcpyDeviceToHost);

    std::cout << "c[0] = " << h_c[0] << std::endl;
    std::cout << "c[1] = " << h_c[1] << std::endl;
    std::cout << "c[2] = " << h_c[2] << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}