cmake_minimum_required(VERSION 3.10)

project(MatrixMultiplication)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(MatrixMultiplication "matrix.h" "matrix.cpp" "main.cpp")


target_compile_options(MatrixMultiplication PRIVATE -fopenmp)
target_link_libraries(MatrixMultiplication PRIVATE -fopenmp)
