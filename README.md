# cuda-benchmark
Describe and compare the speed between CPU and GPU


1. Folder vector_addition: CPU and GPU benchmark
    - I use CPP and CUDA to compare the speed between CPU and GPU with the size of array is 1 billion. My laptop has a CPU and a GPU like images in the folder Result.
    - After coding and running, I recieved the result that GPU is about 7 times faster than CPU when N = 1 billion.
    - In conclusion, when N is small, CPU is faster than GPU because CPU has many cores that can compute parellel for complex task. When N is big, GPU is faster because of the parallel computing of GPU.

2. Folder matrix_multiplication: Naive vs Tiled Shared Memory
    - CPU Naive Matrix Multiplication: use Numpy on CPU
    - GPU Naive Matrix Multiplication: use PyTorch on GPU
    - GPU Tiled Shared Memory Matrix Multiplication: use PyTorch/CUDA on GPU
    - When I run run_benchmark.py, I see the following results:
        + The execution time on the CPU will be very slow, especially with large matrices (e.g. 2048x2048): 
        + The execution time on the GPU (Naive) will be significantly faster, proving that the GPU is superior in parallel processing: 
        + The execution time on the GPU (Tiled Shared Memory) will be faster than the GPU Naive, which is evidence that the optimization using shared memory helps reduce I/O and speed up the computation: 