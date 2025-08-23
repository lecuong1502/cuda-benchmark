# Basic matrix multiplication on CPU using Numpy

import numpy as np
import time

def naive_matpul_cpu(A, B):
    return np.matmul(A, B)

def benchmark_cpu(A, B):
    # Measure the calculating time
    start_time = time.time()
    C = naive_matpul_cpu(A, B)
    end_time = time.time()
    return end_time - start_time