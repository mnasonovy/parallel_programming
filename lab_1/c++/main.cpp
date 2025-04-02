#include "matrix.h"
#include <iostream>


int main() {
    std::string folder = "C:/Users/Uniqu/Desktop/parallel_programming/lab_1/matrix";
    batchMultiplyAndSave(folder);
    std::cout << "Matrix multiplication results saved in: " << folder << std::endl;
    return 0;
}
