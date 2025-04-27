#ifndef CUDA_MULTIPLY_H
#define CUDA_MULTIPLY_H

void multiply_matrices_cuda(const int* A, const int* B, int* C, int N, int threads_x, int threads_y);

#endif
