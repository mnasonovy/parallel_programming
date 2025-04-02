#include "matrix.h"
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>

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

void multiplyAndPrintToFile(const Matrix& A, const Matrix& B, const std::string& filename) {
    int size = A.getSize();
    std::vector<std::vector<int>> result(size, std::vector<int>(size, 0));

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                result[i][j] += A.getData()[i][k] * B.getData()[k][j];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file!" << std::endl;
        return;
    }

    file << "Matrix Size: " << size << "x" << size << "\n";
    file << "Multiplication Time: " << elapsed.count() << " seconds\n\n";

    for (const auto& row : result) {
        for (int val : row) {
            file << val << " ";
        }
        file << std::endl;
    }
    file.close();
}

void batchMultiplyAndSave(const std::string& folder) {
    fs::create_directories(folder);

    for (int size = 10; size <= 1000; size += 10) {
        Matrix A(size);
        Matrix B(size);


        std::string subfolder = folder + "/" + std::to_string(size) + "x" + std::to_string(size);
        fs::create_directory(subfolder);

        std::string firstMatrixFilename = subfolder + "/first_matrix.txt";
        A.printToFile(firstMatrixFilename);

        std::string secondMatrixFilename = subfolder + "/second_matrix.txt";
        B.printToFile(secondMatrixFilename);

        std::string resultFilename = subfolder + "/result.txt";
        multiplyAndPrintToFile(A, B, resultFilename);
    }
}
