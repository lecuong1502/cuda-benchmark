# import pycuda.driver as cuda
# import pycuda.autoinit
# from pycuda.compiler import SourceModule
# import numpy as np
# import time

# # CUDA source
# cuda_code = """
# __global__ void naive_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
#     int row = blockIdx.y * blockDim.y + threadIdx.y;
#     int col = blockIdx.x * blockDim.x + threadIdx.x;
#     if (row < M && col < N) {
#         float Pvalue = 0;
#         for (int i = 0; i < K; ++k) {
#             Pvalue += A[row * K + i] * B[i * N + col];
#         }
#         C[row * N + col] = Pvalue;
#     }
# }
# """

# mod = SourceModule(cuda_code)
# naive_matmul_kernel = mod.get_function("naive_matmul_kernel")

# def benchmark_cuda_naive(size):
#     # Create a matrix on CPU
#     A_cpu = np.random.randn(size, size).astype(np.float32)
#     B_cpu = np.random.randn(size, size).astype(np.float32)
#     C_gpu = np.zeros((size, size), dtype=np.float32)

#     # Allocate the memory on GPU
#     d_A = cuda.mem_alloc(A_cpu.nbytes)
#     d_B = cuda.mem_alloc(B_cpu.nbytes)
#     d_C = cuda.mem_alloc(C_gpu.nbytes)

#     # Copy the data from CPU to GPU
#     cuda.memcpy_htod(d_A, A_cpu)
#     cuda.memcpy_htod(d_B, B_cpu)

#     # Grid and block
#     TILE_SIZE = 16
#     block_size = (TILE_SIZE, TILE_SIZE, 1)
#     grid_size_x = (size + block_size[0] - 1) // block_size[0]
#     grid_size_y = (size + block_size[1] - 1) // block_size[1]
#     grid_size = (grid_size_x, grid_size_y)

#     start_time = time.time()

#     # Call the kernel
#     naive_matmul_kernel (
#         d_A, d_B, d_C,
#         np.int32(size), np.int32(size), np.int32(size),
#         block=block_size, grid=grid_size
#     )

#     # Synchronization and timing
#     cuda.Context.synchronize()
#     end_time = time.time()

#     print(f"Processing time on CUDA Naive: {end_time - start_time: .4f} seconds")

# if __name__ == '__main__':
#     benchmark_cuda_naive(1024)



import torch
import time
def naive_matmul_gpu(A, B):
    """
    Naive matrix multiplication on GPU 
    """
    return torch.matmul(A, B)

def benchmark_gpu(A_gpu, B_gpu):
    """
    Measure the calculating time
    """

    torch.cuda.synchronize() # Synchronize the GPU
    start_time = time.time()
    C = naive_matmul_gpu(A_gpu, B_gpu)

    torch.cuda.synchronize() # Synchronize the GPU
    end_time = time.time()
    return end_time - start_time