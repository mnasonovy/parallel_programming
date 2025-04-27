#include "cuda_multiply.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMultiplyKernel(const int* A, const int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// 🛠 Теперь multiply_matrices_cuda принимает размеры блока как аргументы
void multiply_matrices_cuda(const int* A, const int* B, int* C, int N, int threads_x, int threads_y) {
    int size = N * N * sizeof(int);

    int* d_A;
    int* d_B;
    int* d_C;

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // ➡️ Теперь создаем сетку из переданных параметров
    dim3 threadsPerBlock(threads_x, threads_y);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiplyKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
