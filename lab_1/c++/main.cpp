#include "matrix.h"
#include <iostream>


int main() {
    std::string folder = "S:/3rd_cource/parallel_programming/lab_1/matrix";
    batchMultiplyAndSave(folder);
    std::cout << "Matrix multiplication results saved in: " << folder << std::endl;
    return 0;
}
