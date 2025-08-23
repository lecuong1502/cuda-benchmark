import numpy as np
import torch
import time

from data_generator import create_matrices
from naive_cpu import benchmark_cpu
from naive_gpu import benchmark_gpu
from tiled_gpu import compile_and_run_tiled_matmul, benchmark_tiled_gpu

def run_all_benchmarks(size):
    print(f"Start benchmark with the matrix {size}x{size}...")

    (A_cpu, B_cpu), (A_gpu, B_gpu) = create_matrices(size)
    
    # Benchmark CPU
    cpu_time = benchmark_cpu(A_cpu, B_cpu)
    print (f"Time on CPU (Numpy): {cpu_time: .4f} seconds")

    # Benchmark GPU Naive
    if A_gpu is not None:
        naive_gpu_time = benchmark_gpu(A_gpu, B_gpu)
        print(f"Time on GPU (Pytorch Naive): {naive_gpu_time: .4f} seconds")

        # Benchmark GPU Tiled
        tiled_kernel = compile_and_run_tiled_matmul()
        tiled_gpu_time = benchmark_tiled_gpu(A_gpu, B_gpu, size, size, size, tiled_kernel)
        print(f"Time on GPU (Tiled Shared Memory): {tiled_gpu_time: .4f} seconds")

        # Print the result
        print("\n--- Performance comparison ---")
        print(f"GPU Naive is {cpu_time / naive_gpu_time:.2f} times faster than CPU.")
        print(f"GPU Tiled is {cpu_time / tiled_gpu_time:.2f} times faster than CPU.")
        print(f"GPU Tiled is {naive_gpu_time / tiled_gpu_time:.2f} times faster than GPU Naive")

    else:
        print("Cannot find the GPU, only run the benchmark on CPU")

if __name__ == '__main__':
    sizes = [512, 1024, 2048]
    for s in sizes:
        run_all_benchmarks(s)
        print("-" * 40)