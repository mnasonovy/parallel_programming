import os
import numpy as np
import json

def read_matrix_from_file(filename, skip_lines=0):
    with open(filename, 'r') as file:
        lines = file.readlines()[skip_lines:]
        matrix_data = [line.split() for line in lines if line.strip() != ""]

    row_lengths = [len(row) for row in matrix_data]
    if len(set(row_lengths)) != 1:
        raise ValueError(f"Матрица в файле {filename} имеет строки разной длины!")

    return np.array(matrix_data, dtype=int)


def compare_matrices(matrix1, matrix2):
    return matrix1.shape == matrix2.shape and np.array_equal(matrix1, matrix2)

def check_matrix_multiplication(base_folder, start_size=10, step=10):
    thread_folders = [d for d in os.listdir(base_folder) if d.startswith("threads_")]
    thread_folders.sort(key=lambda x: int(x.split('_')[1]))

    for thread_folder in thread_folders:
        path_to_thread = os.path.join(base_folder, thread_folder)
        subfolders = sorted(os.listdir(path_to_thread), key=lambda x: int(x.split('x')[0]))

        for sub in subfolders:
            sub_path = os.path.join(path_to_thread, sub)
            first = os.path.join(sub_path, 'first_matrix.txt')
            second = os.path.join(sub_path, 'second_matrix.txt')
            result = os.path.join(sub_path, 'result.txt')

            try:
                A = read_matrix_from_file(first)
                B = read_matrix_from_file(second)
                expected = read_matrix_from_file(result, skip_lines=3)

                actual = np.dot(A, B)
                if not compare_matrices(actual, expected):
                    raise ValueError(f"Несовпадение результатов в {sub_path}")

            except Exception as e:
                print(f"❌ Ошибка в {sub_path}: {e}")
                continue

        print(f"✅ Проверка матриц, вычисленных {thread_folder.replace('_', ' ')} завершена успешно!")

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config['start_size'], config['step'], config['base_folder']

if __name__ == "__main__":
    config_file = "S:/3rd_cource/parallel_programming/lab_2/python/config.json"
    start_size, step, base_folder = load_config(config_file)

    try:
        check_matrix_multiplication(base_folder, start_size, step)
    except ValueError as e:
        print(f"Ошибка: {e}")
