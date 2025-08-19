# cuda-benchmark
Describe and compare the speed between CPU and GPU


1. Folder vector_addition: CPU and GPU benchmark
    - I use CPP and CUDA to compare the speed between CPU and GPU with the size of array is 1 billion. My laptop has a CPU and a GPU like images in the folder Result.
    - After coding and running, I recieved the result that GPU is about 7 times faster than CPU when N = 1 billion.
    - In conclusion, when N is small, CPU is faster than GPU because CPU has many cores that can compute parellel for complex task. When N is big, GPU is faster because of the parallel computing of GPU.

2. Folder matrix_multiplication: Naive vs Tiled Shared Memory

Result of benchmark image: