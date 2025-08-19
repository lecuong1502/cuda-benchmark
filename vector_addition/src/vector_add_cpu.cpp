// Addition in C++ using the CPU

#include <vector>
using namespace std;

void vector_add_cpu(const vector<float>& A, const vector<float>& B, vector<float>& C){
    size_t size = A.size();
    for (size_t i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }
}