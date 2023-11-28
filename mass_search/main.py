import csv
import time

import numpy as np
from matplotlib import pyplot as plt
from numba import cuda

# Функция для поиска на CPU
def mass_search_CPU(R, N, H):
    for j in range(R.shape[1]):
        for i in range(R.shape[0]):
            n = N[i]
            for k in range(len(n)):
                # Используем векторизацию для ускорения вычислений
                R[i, j - k] -= np.sum(n[k] == H[j])

    return R

# Функция для поиска на GPU
@cuda.jit
def mass_search_GPU(R, N, H):
    i, j = cuda.grid(2)
    if i < R.shape[0] and j < R.shape[1]:
        n = N[i]
        for k in range(len(n)):
            for p in range(len(H[j])):
                if n[k] == H[j][p]:
                    R[i, j - k] -= 1

# Сохранение массива в CSV файл
def save_array_to_csv(my_array, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerows(my_array)

# Основная функция для запуска вычислений
def start_calculation(sizes: np.ndarray):
    table_values_CPU = np.zeros((0, sizes.shape[0]))
    table_values_GPU = np.zeros((0, sizes.shape[0]))

    for n in range(10):
        array_time_CPU = []
        array_time_GPU = []

        for size in sizes:
            # Генерация случайных данных
            N = np.random.randint(len(ABC), size=(size, 3), dtype=np.uint8)
            H = np.random.randint(len(ABC), size=(size, 3), dtype=np.uint8)
            R = np.zeros((size, 3), dtype=int)

            # Измерение времени выполнения на CPU
            start_CPU = time.time()
            mass_search_CPU(R, N, H)
            end_CPU = time.time()
            execution_CPU = end_CPU - start_CPU
            array_time_CPU.append(execution_CPU)

            # Измерение времени выполнения на GPU
            start_GPU = time.time()
            R_GPU = cuda.to_device(R)
            N_GPU = cuda.to_device(N)
            H_GPU = cuda.to_device(H)

            mass_search_GPU[blocks_per_grid, threads_per_block](R_GPU, N_GPU, H_GPU)

            cuda.synchronize()
            R_GPU.copy_to_host(R)

            end_GPU = time.time()
            execution_GPU = end_GPU - start_GPU
            array_time_GPU.append(execution_GPU)
            save_array_to_csv(R, f"GPU_{size}")

        # Собираем результаты выполнения в таблицу
        table_values_CPU = np.vstack((table_values_CPU, np.array(array_time_CPU).reshape((1, 10))))
        table_values_GPU = np.vstack((table_values_GPU, np.array(array_time_GPU).reshape((1, 10))))

    table_values_CPU = np.squeeze(table_values_CPU)
    table_values_GPU = np.squeeze(table_values_GPU)

    # Вычисляем среднее время выполнения для каждого размера данных
    mas_CPU_time = [np.mean(table_values_CPU[:, i]) for i in range(table_values_CPU.shape[1])]
    mas_GPU_time = [np.mean(table_values_GPU[:, i]) for i in range(table_values_GPU.shape[1])]

    return mas_CPU_time, mas_GPU_time

# Основная часть программы
if __name__ == "__main__":
    # Параметры для GPU вычислений
    threads_per_block = (8, 8)
    blocks_per_grid = (16, 16)

    # Размеры данных
    sizes = np.linspace(160, 1600, 10, dtype=int)

    ABC = np.arange(256)

    # Запуск вычислений
    mas_CPU_time, mas_GPU_time = start_calculation(sizes)
    mas_CPU_time = np.array(mas_CPU_time)
    mas_GPU_time = np.array(mas_GPU_time)

    # Вывод результатов
    print(f"Время CPU : \n{mas_CPU_time}\n")
    print(f"Время GPU : \n{mas_GPU_time}\n")

    # Вывод ускорения
    print(f"Ускорение GPU отнисительно CPU: {mas_CPU_time / mas_GPU_time}")

    # Построение графика ускорения
    plt.grid()
    plt.plot(sizes, mas_CPU_time / mas_GPU_time, label='Ускорение')

    plt.title('Графики ускорения GPU относительно CPU')
    plt.xlabel('Размерность')
    plt.ylabel('Ускорение')
    plt.legend()
    plt.show()
