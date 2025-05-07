#!/bin/bash

folder="/mnt/s/3rd_cource/parallel_programming/lab_3/matrix"

for np in 1 2 4 8 16
do
    echo "Running with $np processes..."
    mpirun --oversubscribe -np $np /mnt/s/3rd_cource/parallel_programming/lab_3/c++/cmake-build-debug-wsl/MatrixMultiplicationMPI "$folder"
done
