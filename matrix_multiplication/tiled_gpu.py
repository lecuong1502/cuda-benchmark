# Kernel CUDA to use the shared memory.
# This technical decreases the access to global memory (very slow) and use shared memory (very fast)

# Shared memory: is the general memory of a block thread on GPU

import torch
from torch.utils.cpp_extension import load_inline
import time 

cuda_source = """
#include <cuda.h>
__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // The size of tile
    const int TILE_SIZE = 32;

    // Shared memory
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // The position of thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float Pvalue = 0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Upload the data from global memory to shared memory
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + (t * TILE_SIZE + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();

        // Calculate on the shared memory
        for (int i = 0; i < TILE_SIZE; ++i) {
            Pvalue += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = Pvalue;
    }
}
"""

def compile_and_run_tiled_matmul():
    tiled_matmul = load_inline(
        name='tiled_matmul',
        cpp_sources='',
        cuda_sources=cuda_source,
        functions=['tiled_matmul_kernel'],
        verbose=True,
    )
    return tiled_matmul.tiled_matmul_kernel

def benchmark_tiled_gpu(A_gpu, B_gpu, M, N, K, kernel_func):
    C_gpu = torch.zeros(M, N, device='cuda')

    # Block and Grid
    TILE_SIZE = 32
    block_size = (TILE_SIZE, TILE_SIZE)
    grid_size_x = (N + block_size[0] - 1) // block_size[0]
    grid_size_y = (M + block_size[1] - 1) // block_size[1]
    grid_size = (grid_size_x, grid_size_y)

    torch.cuda.synchronize()
    start_time = time.time()

    # Call the kernel
    kernel_func(A_gpu, B_gpu, C_gpu, M, N, K, grid=grid_size, block=block_size)

    torch.cuda.synchronize()
    end_time = time.time()
    return end_time - start_time