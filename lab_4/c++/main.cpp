#include "matrix.h"
#include <iostream>


int main() {
    std::string folder = "matrix";
    batchMultiplyAndSave(folder);
    std::cout << "Matrix multiplication results saved in: " << folder << std::endl;
    return 0;
}
