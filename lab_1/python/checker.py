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

def write_matrix_to_file(matrix, filename):
    with open(filename, 'w') as file:
        for row in matrix:
            file.write(' '.join(map(str, row)) + '\n')

def compare_matrices(matrix1, matrix2):
    return matrix1.shape == matrix2.shape and np.array_equal(matrix1, matrix2)

def check_matrix_multiplication(base_folder, start_size=5, step=5):
    folder_names = sorted(os.listdir(base_folder), key=lambda x: int(x.split('x')[0]) if x.split('x')[0].isdigit() else 0)
    all_checks_passed = True

    for folder_name in folder_names:
        folder_path = os.path.join(base_folder, folder_name)
        try:
            size = int(folder_name.split('x')[0])
            if size < start_size or (size - start_size) % step != 0:
                continue
        except ValueError:
            continue

        first_matrix_file = os.path.join(folder_path, 'first_matrix.txt')
        second_matrix_file = os.path.join(folder_path, 'second_matrix.txt')
        result_file = os.path.join(folder_path, 'result.txt')

        if os.path.exists(first_matrix_file) and os.path.exists(second_matrix_file) and os.path.exists(result_file):
            try:
                A = read_matrix_from_file(first_matrix_file)
                B = read_matrix_from_file(second_matrix_file)
                expected_result = read_matrix_from_file(result_file, skip_lines=3)
            except ValueError as e:
                raise ValueError(f"Ошибка при чтении файла в папке {folder_name}: {e}")
            try:
                result = np.dot(A, B)
                if not compare_matrices(result, expected_result):
                    raise ValueError(f"Ошибка в папке {folder_name}: результаты не совпадают!")
            except ValueError as e:
                raise ValueError(f"Ошибка при умножении матриц в папке {folder_name}: {e}")

    print("✅ Все проверки завершены успешно!")

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config['start_size'], config['step'], config['base_folder']

if __name__ == "__main__":
    config_file = "S:/3rd_cource/parallel_programming/lab_1/python/config.json"
    start_size, step, base_folder = load_config(config_file)

    try:
        check_matrix_multiplication(base_folder, start_size, step)
    except ValueError as e:
        print(f"Ошибка: {e}")
