#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Define kernel CUDA for naive matrix multiplication
__global__ void naive_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float Pvalue = 0;
        for (int i = 0; i < K; ++i) {
            Pvalue += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = Pvalue;
    }
}

// Wrapper function to call kernel from CPU
void naive_matmul_gpu(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    const int TILE_SIZE = 16;
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Call the kernel
    naive_matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}