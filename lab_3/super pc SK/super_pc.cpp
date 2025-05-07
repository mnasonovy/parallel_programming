#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

class Matrix {
private:
    std::vector<std::vector<int>> data;
    int size;

public:
    Matrix(int s) : size(s) {
        data.resize(size, std::vector<int>(size));
        std::srand(static_cast<unsigned int>(std::time(0)));
        for (int i = 0; i < size; ++i)
            for (int j = 0; j < size; ++j)
                data[i][j] = 1 + std::rand() % 100;
    }

    int getSize() const { return size; }
    const std::vector<std::vector<int>>& getData() const { return data; }
};

void multiplyAndReportTime(const Matrix& A, const Matrix& B, int num_procs) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = A.getSize();
    int rows_per_proc = N / num_procs;
    int remainder = N % num_procs;

    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    int start_row = rank * rows_per_proc + std::min(rank, remainder);

    std::vector<std::vector<int>> local_result(local_rows, std::vector<int>(N, 0));

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k)
                local_result[i][j] += A.getData()[start_row + i][k] * B.getData()[k][j];

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::vector<int> sendcounts(num_procs), displs(num_procs);
    int offset = 0;
    for (int i = 0; i < num_procs; ++i) {
        sendcounts[i] = (rows_per_proc + (i < remainder ? 1 : 0)) * N;
        displs[i] = offset;
        offset += sendcounts[i];
    }

    std::vector<int> global_result;
    if (rank == 0) global_result.resize(N * N);

    std::vector<int> local_flat(local_rows * N);
    for (int i = 0; i < local_rows; ++i)
        for (int j = 0; j < N; ++j)
            local_flat[i * N + j] = local_result[i][j];

    MPI_Gatherv(local_flat.data(), local_rows * N, MPI_INT,
                global_result.data(), sendcounts.data(), displs.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Matrix Size: " << N << "x" << N << "\n";
        std::cout << "Multiplication Time: " << elapsed.count() << " seconds\n";
        std::cout << "Threads: " << num_procs << "\n\n";
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    for (int size = 500; size <= 5000; size += 500) {
        Matrix A(size);
        Matrix B(size);
        multiplyAndReportTime(A, B, num_procs);
        MPI_Barrier(MPI_COMM_WORLD); // Синхронизация перед следующей задачей
    }

    MPI_Finalize();
    return 0;
}
