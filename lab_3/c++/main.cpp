#include "matrix.h"
#include <mpi.h>
#include <iostream>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    if (argc < 2) {
        if (int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank), rank == 0)
            std::cerr << "Ошибка: укажите путь к папке для сохранения результатов!" << std::endl;
        MPI_Finalize();
        return 1;
    }

    std::string folder = argv[1];
    batchMultiplyAndSave(folder);

    MPI_Finalize();
    return 0;
}
