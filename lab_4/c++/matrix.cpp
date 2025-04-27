// matrix.cpp (финальный чистый вариант для CUDA)

#include "matrix.h"
#include "../cuda/cuda_multiply.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    data.resize(rows, std::vector<int>(cols, 0));
}

void Matrix::fillRandom() {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            data[i][j] = rand() % 100;
}

void Matrix::saveToFile(const std::string& filename) const {
    std::ofstream file(filename);
    for (const auto& row : data) {
        for (const auto& elem : row) {
            file << elem << " ";
        }
        file << "\n";
    }
}

Matrix Matrix::loadFromFile(const std::string& filename) {
    std::ifstream file(filename);
    std::vector<std::vector<int>> temp_data;
    int value;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::vector<int> row;
        while (iss >> value) {
            row.push_back(value);
        }
        if (!row.empty())
            temp_data.push_back(row);
    }

    int rows = temp_data.size();
    int cols = temp_data.empty() ? 0 : temp_data[0].size();

    Matrix matrix(rows, cols);
    matrix.data = std::move(temp_data);
    return matrix;
}

void multiplyAndPrintToFile(const Matrix& A, const Matrix& B, const std::string& output_file, int threads_per_dim) {
    int N = A.rows;
    std::vector<int> flat_A(N * N);
    std::vector<int> flat_B(N * N);
    std::vector<int> flat_C(N * N, 0);

    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            flat_A[i * N + j] = A.data[i][j];
            flat_B[i * N + j] = B.data[i][j];
        }

    // CUDA warm-up
    multiply_matrices_cuda(flat_A.data(), flat_B.data(), flat_C.data(), N, threads_per_dim, threads_per_dim);
    cudaDeviceSynchronize();

    // Чистое измерение времени
    auto start = std::chrono::high_resolution_clock::now();
    multiply_matrices_cuda(flat_A.data(), flat_B.data(), flat_C.data(), N, threads_per_dim, threads_per_dim);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::ofstream file(output_file);
    file << "Matrix Size: " << N << "x" << N << "\n";
    file << "Multiplication Time: " << duration.count() << " seconds\n";
    file << "Threads: " << threads_per_dim << "\n";

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            file << flat_C[i * N + j] << " ";
        }
        file << "\n";
    }
}

void batchMultiplyAndSave(const std::string& base_path) {
    int start_size = 500;
    int step = 500;
    int max_size = 5000;

    for (int threads_per_dim = 1; threads_per_dim <= 16; threads_per_dim *= 2) {
        for (int size = start_size; size <= max_size; size += step) {
            std::ostringstream folder_path;
            folder_path << base_path << "/threads_" << threads_per_dim << "/" << size << "x" << size;

            fs::create_directories(folder_path.str());

            Matrix A(size, size);
            Matrix B(size, size);

            A.fillRandom();
            B.fillRandom();

            A.saveToFile(folder_path.str() + "/first_matrix.txt");
            B.saveToFile(folder_path.str() + "/second_matrix.txt");

            multiplyAndPrintToFile(A, B, folder_path.str() + "/result.txt", threads_per_dim);
        }
    }
}
