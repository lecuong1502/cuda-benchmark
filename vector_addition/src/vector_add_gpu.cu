// Addition in C++ using the CPU

#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../include/timer.h"
using namespace std;

// CUDA kernel to add two vectors
__global__ void vector_add_kernel(const float* A, const float* B, float* C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) C[i] = A[i] + B[i];
}

void vector_add_gpu(const vector<float>& A, const vector<float>& B, vector<float>& C) {
    size_t size = A.size();
    float *d_A, *d_B, *d_C;

    // Allocate memory on GPU
    cudaMalloc(&d_A, size * sizeof(float));
    cudaMalloc(&d_B, size * sizeof(float));
    cudaMalloc(&d_C, size * sizeof(float));

    // Copy the data from CPU to GPU
    cudaMemcpy(d_A, A.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    // Configure number of blocks and threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    // Run CUDA kernel
    vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);

    // Copy the result from GPU to CPU
    cudaMemcpy(C.data(), d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free the memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}