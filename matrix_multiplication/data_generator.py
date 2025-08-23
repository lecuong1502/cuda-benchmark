# Create big matrices to check

import numpy as np
import torch

def create_matrices(size):
    # Create 2 random matrices A and B with the size (size, size)
    if torch.cuda.is_available():
        # Create a matrix on GPU
        A_gpu = torch.randn(size, size, device='cuda')
        B_gpu = torch.randn(size, size, device='cuda')

    else:
        A_gpu, B_gpu = None, None

    # Create a matrix on CPU
    A_cpu = np.random.randn(size, size)
    B_cpu = np.random.randn(size, size)

    return (A_cpu, B_cpu), (A_gpu, B_gpu)

if __name__ == '__main__':
    cpu_mats, gpu_mats = create_matrices(1024)
    print("The size of matrix on CPU: ", cpu_mats[0].shape)
    if gpu_mats[0] is not None:
        print("The size of matrix on GPU: ", gpu_mats[0].shape)