#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

class Matrix {
public:
    int rows;
    int cols;
    std::vector<std::vector<int>> data;

    Matrix(int rows, int cols);
    void fillRandom();
    void saveToFile(const std::string& filename) const;
    static Matrix loadFromFile(const std::string& filename);
};

void multiplyAndPrintToFile(const Matrix& A, const Matrix& B, const std::string& filename, int threads);
void batchMultiplyAndSave(const std::string& folder);

#endif // MATRIX_H
