import numpy as np
from numba import cuda
from time import time
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import pandas as pd
import matplotlib.pyplot as plt

# число потоков на блок
TPB = 16
# количество итераций для вычисления π
pi_size = 32
# количество блолков на сетку
BPG = int(pi_size / TPB)


# Функция подсчета значения Пи на CPU
def CPU_calc(pi_size, N):
    res = np.zeros(pi_size)
    for i in range(pi_size):
        x = np.random.uniform(size=N)
        y = np.random.uniform(size=N)
        z = x ** 2 + y ** 2 <= 1
        res[i] = 4.0 * sum(z) / N
    return res


# Ядро для GPU
@cuda.jit
def pi_calcul(res, rng):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    h = 0
    # - Возвращает случайное значение от 0. до 1.
    if i < len(rng):
        x = xoroshiro128p_uniform_float32(rng, i)
        y = xoroshiro128p_uniform_float32(rng, i)
        if x ** 2 + y ** 2 <= 1.0:
            h += 1
    cuda.atomic.add(res, 0, h)


# Функция, выполняющая вычисление значения Пи на GPU
def GPU_calc(N):
    # - для генерации случайного массива на CUDA
    rng_states = create_xoroshiro128p_states(N, seed=1)
    res = np.zeros(1)

    dev_res = cuda.to_device(res)
    pi_calcul[N, TPB](dev_res, rng_states)
    tmp_ = dev_res.copy_to_host()
    pi_ = 4 * tmp_[0] / N
    return pi_


# 10 итераций подсчета числа Пи с увеличением значения N на 10000 после каждой итерации
iteration_count = np.arange(1, 11, 1)
N = 10000

df = {"Время на GPU": np.zeros(len(iteration_count), dtype=float),
      "Время на CPU": np.zeros(len(iteration_count), dtype=float),
      "Пи на GPU": np.zeros(len(iteration_count), dtype=float),
      "Пи на CPU": np.zeros(len(iteration_count), dtype=float),
      "Ускорение": np.zeros(len(iteration_count), dtype=float),
      "Кол-во точек": np.zeros(len(iteration_count), dtype=int)}

df = pd.DataFrame(df, index=iteration_count)
for i in iteration_count:
    cpu_start = time()
    cpu_pi = CPU_calc(pi_size, N).mean()
    cpu_time = time() - cpu_start

    gpu_start = time()
    gpu_pi = GPU_calc(N)
    gpu_time = time() - gpu_start

    df.loc[i, "Время на GPU"] = gpu_time
    df.loc[i, "Время на CPU"] = cpu_time
    df.loc[i, "Пи на GPU"] = gpu_pi
    df.loc[i, "Пи на CPU"] = cpu_pi
    df["Ускорение"] = df["Время на CPU"] / df["Время на GPU"]
    df.loc[i, "Кол-во точек"] = N

    N += 10000

print(df)
plt.plot(df["Кол-во точек"], df["Ускорение"], label='Ускорение GPU относительно CPU', linestyle='-', marker='')
plt.title('Ускорение времени GPU относительно CPU')
plt.xlabel('Количество точек (N)')
plt.ylabel('Ускорение')
plt.legend()
plt.grid()
plt.show()
