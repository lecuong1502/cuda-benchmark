#include <iostream>
#include <vector>
#include <random>
#include "../include/timer.h"

using namespace std;

// Functions from other files
void vector_add_cpu(const vector<float>& A, const vector<float>& B, vector<float>& C);
void vector_add_gpu(const vector<float>& A, const vector<float>& B, vector<float>& C);

int main() {
    const int N = 1000000000; // Choose the big size to see the difference clearly
    vector<float> A(N), B(N), C_cpu(N), C_gpu(N);
    
    // Allocate the random data
    mt19937 gen(42);                                    // Mersense Twister algorithm with seed = 42
    uniform_real_distribution<float> dis(0.0f, 100.0f);  // Uniform distribution
    for (int i = 0; i < N; ++i) {
        A[i] = dis(gen);
        B[i] = dis(gen);
    }

    cout << "Size of vector: " << N << endl;

    // Benchmark CPU
    {
        Timer timer("CPU");
        vector_add_cpu(A, B, C_cpu);
    }

    // Benchmark GPU
    {
        Timer timer("GPU");
        vector_add_gpu(A, B, C_gpu);
    }

    return 0;
}