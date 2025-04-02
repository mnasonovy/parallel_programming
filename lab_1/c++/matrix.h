#ifndef MATRIX_H
#define MATRIX_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>

class Matrix {
private:
    std::vector<std::vector<int>> data;
    int size;

public:
    Matrix(int s);
    void printToFile(const std::string& filename) const;
    int getSize() const;
    const std::vector<std::vector<int>>& getData() const;
};

void multiplyAndPrintToFile(const Matrix& A, const Matrix& B, const std::string& filename);
void batchMultiplyAndSave(const std::string& folder);
#endif 
