import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import mplcyberpunk

def extract_matrix_size_and_time(result_file):
    with open(result_file, 'r') as file:
        lines = file.readlines()
        size_line = lines[0].strip()
        time_line = lines[1].strip()

    size = size_line.split(':')[1].strip()
    rows, _ = map(int, size.split('x'))

    time_str = time_line.split(':')[1].strip()
    time_taken = float(time_str.split()[0])
    return rows, time_taken

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config['start_size'], config['step'], config['base_folder']

def plot_parallel_performance(base_folder, start_size, step):
    plt.style.use("cyberpunk")
    plt.figure(figsize=(14, 10))

    thread_folders = sorted(
        [d for d in os.listdir(base_folder) if d.startswith("threads_")],
        key=lambda x: int(x.split('_')[1])
    )

    for folder in thread_folders:
        threads = int(folder.split('_')[1])
        thread_path = os.path.join(base_folder, folder)
        sizes = []
        times = []

        for subfolder in sorted(os.listdir(thread_path), key=lambda x: int(x.split('x')[0])):
            try:
                size = int(subfolder.split('x')[0])
                if size < start_size or (size - start_size) % step != 0:
                    continue
            except ValueError:
                continue

            result_file = os.path.join(thread_path, subfolder, "result.txt")
            if os.path.exists(result_file):
                try:
                    matrix_size, time = extract_matrix_size_and_time(result_file)
                    sizes.append(matrix_size)
                    times.append(time)
                except Exception as e:
                    print(f"Ошибка при чтении {result_file}: {e}")

        if sizes and times:
            if threads == 1:
                label = "1 поток"
            elif threads in [2, 3, 4]:
                label = f"{threads} потока"
            else:
                label = f"{threads} потоков"

            plt.plot(sizes, times, marker='o', linewidth=2, label=label)

    plt.title('Сравнение времени выполнения при разном количестве потоков', fontsize=18, fontweight='bold')
    plt.xlabel('Размер матрицы (n x n)', fontsize=14)
    plt.ylabel('Время выполнения (секунды)', fontsize=14)
    plt.legend(title='Кол-во потоков', fontsize=12, title_fontsize=13)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    mplcyberpunk.add_glow_effects()
    plt.tight_layout()
    plt.savefig("S:/3rd_cource/parallel_programming/lab_1/python/performance_plot.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    config_file = "S:/3rd_cource/parallel_programming/lab_4/python/config.json"
    start_size, step, base_folder = load_config(config_file)

    plot_parallel_performance(base_folder, start_size, step)
