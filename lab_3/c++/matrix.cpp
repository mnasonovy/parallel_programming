#include "matrix.h"
#include <mpi.h>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

Matrix::Matrix(int s) : size(s) {
    data.resize(size, std::vector<int>(size));
    std::srand(static_cast<unsigned int>(std::time(0)));
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            data[i][j] = 1 + std::rand() % 100;
        }
    }
}

void Matrix::printToFile(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }
    for (const auto& row : data) {
        for (int val : row) {
            file << val << " ";
        }
        file << std::endl;
    }
    file.close();
}

int Matrix::getSize() const {
    return size;
}

const std::vector<std::vector<int>>& Matrix::getData() const {
    return data;
}

void multiplyAndPrintToFile(const Matrix& A, const Matrix& B, const std::string& filename, int num_procs) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = A.getSize();
    int rows_per_proc = N / num_procs;
    int remainder = N % num_procs;

    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int start_row = rank * rows_per_proc + std::min(rank, remainder);

    std::vector<std::vector<int>> local_result(local_rows, std::vector<int>(N, 0));

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                local_result[i][j] += A.getData()[start_row + i][k] * B.getData()[k][j];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::vector<int> sendcounts(num_procs);
    std::vector<int> displs(num_procs);

    int offset = 0;
    for (int i = 0; i < num_procs; ++i) {
        sendcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * N;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    std::vector<int> global_result;
    if (rank == 0) {
        global_result.resize(N * N);
    }

    std::vector<int> local_result_flat(local_rows * N);
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < N; ++j) {
            local_result_flat[i * N + j] = local_result[i][j];
        }
    }

    MPI_Gatherv(local_result_flat.data(), local_rows * N, MPI_INT,
                global_result.data(), sendcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::ofstream file(filename);
        if (!file) {
            std::cerr << "Error opening file!" << std::endl;
            return;
        }

        file << "Matrix Size: " << N << "x" << N << "\n";
        file << "Multiplication Time: " << elapsed.count() << " seconds\n";
        file << "Threads: " << num_procs << "\n\n";

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                file << global_result[i * N + j] << " ";
            }
            file << "\n";
        }
        file.close();
    }
}

void batchMultiplyAndSave(const std::string& folder) {
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (rank == 0) {
        fs::create_directories(folder + "/threads_" + std::to_string(num_procs));
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int size = 10; size <= 1000; size += 10) {
        Matrix A(size);
        Matrix B(size);

        if (rank == 0) {
            std::string subfolder = folder + "/threads_" + std::to_string(num_procs) + "/" + std::to_string(size) + "x" + std::to_string(size);
            fs::create_directories(subfolder);

            A.printToFile(subfolder + "/first_matrix.txt");
            B.printToFile(subfolder + "/second_matrix.txt");
        }

        MPI_Barrier(MPI_COMM_WORLD);

        std::string resultFilename = folder + "/threads_" + std::to_string(num_procs) + "/" + std::to_string(size) + "x" + std::to_string(size) + "/result.txt";
        multiplyAndPrintToFile(A, B, resultFilename, num_procs);

        MPI_Barrier(MPI_COMM_WORLD);
    }
}
