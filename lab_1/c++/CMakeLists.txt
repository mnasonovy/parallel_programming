cmake_minimum_required(VERSION 3.10)

project(MatrixMultiplication)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(MatrixMultiplication "matrix.h" "matrix.cpp" "main.cpp")
