import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

def extract_matrix_size_and_time(result_file):
    """Извлекает размер матрицы и время выполнения из файла result.txt"""
    with open(result_file, 'r') as file:
        lines = file.readlines()
        size_line = lines[0].strip()  
        time_line = lines[1].strip()  

    size = size_line.split(':')[1].strip()
    rows, cols = map(int, size.split('x'))

    time_str = time_line.split(':')[1].strip()
    time_taken = float(time_str.split()[0])  

    return rows, time_taken

def load_config(config_file):
    """Загружает параметры из JSON файла"""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config['start_size'], config['step']

def plot_performance(base_folder, start_size, step):
    sizes = []
    times = []

    folder_names = [folder_name for folder_name in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, folder_name))]
    folder_names.sort(key=lambda x: int(x.split('x')[0])) 

    for folder_name in folder_names:
        folder_path = os.path.join(base_folder, folder_name)

        try:
            size = int(folder_name.split('x')[0])
            if size < start_size or (size - start_size) % step != 0:
                continue  
        except ValueError:
            continue 

        result_file = os.path.join(folder_path, 'result.txt')

        if os.path.exists(result_file):
            try:
                size, time_taken = extract_matrix_size_and_time(result_file)
                sizes.append(size)
                times.append(time_taken)
            except Exception as e:
                print(f"Ошибка при извлечении данных из {result_file}: {e}")
                continue

    # Настроим стиль Seaborn
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)

    # Создаем график с использованием Seaborn
    plt.figure(figsize=(14, 8))
    sns.lineplot(x=sizes, y=times, marker='o', linewidth=3, markersize=8, color='dodgerblue', linestyle='-', label='Время выполнения')

    # Дополнительные улучшения
    plt.title('Зависимость времени умножения матриц от их размера', fontsize=18, fontweight='bold')
    plt.xlabel('Размер матрицы (n x n)', fontsize=14)
    plt.ylabel('Время выполнения (секунды)', fontsize=14)
    
    # Добавление логарифмической шкалы по оси Y (если нужно)
    plt.yscale('log')

    # Уменьшаем количество подписей на оси X, показываем только через каждый 2-й элемент
    step_size = 2
    plt.xticks(sizes[::step_size], rotation=45, fontsize=12)

    # Форматируем ось Y, чтобы отображались только 2 знака после запятой
    plt.yticks(np.logspace(np.log10(min(times)), np.log10(max(times)), num=10), fontsize=12)
    plt.gca().get_yaxis().set_major_formatter(plt.ScalarFormatter())

    # Включаем сетку
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Показываем график
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    config_file = 'C:/Users/Uniqu/Desktop/parallel_programming/lab_1/python/config.json'  
    start_size, step = load_config(config_file)

    base_folder = 'C:/Users/Uniqu/Desktop/parallel_programming/lab_1/matrix'  
    plot_performance(base_folder, start_size, step)
