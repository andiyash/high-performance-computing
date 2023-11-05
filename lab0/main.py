import numpy as np
import time
from numba import cuda
import math
from matplotlib import pyplot as plt

matrix_size = 100

# Inicialisation of matrixes for CPU
cpu_matrix1 = np.random.randint(0, 10, (matrix_size, matrix_size))
cpu_matrix2 = np.random.randint(0, 10, (matrix_size, matrix_size))
cpu_matrix_res = np.zeros((matrix_size, matrix_size), dtype=int)

# Inicialisation of matrixes for GPU
gpu_matrix1 = cuda.to_device(cpu_matrix1)
gpu_matrix2 = cuda.to_device(cpu_matrix2)
gpu_matrix_res = cuda.device_array((len(cpu_matrix1), len(cpu_matrix2)))


# Function of MatMul on CPU
def cpu_mat_mul(A, B, C):
    for i in range(matrix_size):
        for j in range(matrix_size):
            res = 0
            for k in range(matrix_size):
                res += A[i, k] * B[k, j]
            C[i, j] = res


def cpu_calc():
    print("CPU begin working.")
    start_time = time.time()
    cpu_mat_mul(cpu_matrix1, cpu_matrix2, cpu_matrix_res)
    print("%s seconds is time for calculation on CPU" % (time.time() - start_time))


@cuda.jit
def gpu_mat_mul(A, B, C):
    for i in range(matrix_size):
        for j in range(matrix_size):
            rez = 0
            for z in range(matrix_size):
                rez += A[i, z] * B[z, j]
            C[i, j] = rez


def gpu_calc():
    # Kernel parameters
    # Amount of threads in block
    threadsperblock = (32, 32)
    # Amount of blocks in grid by x
    blockspergrid_x = int(math.ceil(cpu_matrix1.shape[0] / threadsperblock[0]))
    # Amount of blocks in grid by y
    blockspergrid_y = int(math.ceil(cpu_matrix2.shape[1] / threadsperblock[1]))
    # Amount of blocks in whole grid
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    print("Grid size = ", blockspergrid, threadsperblock)
    print("End of CPU work!\n")

    print("GPU started its work...")
    start_time = time.time()
    # Calculation on GPU on given blocks and threads
    gpu_mat_mul[blockspergrid, threadsperblock](gpu_matrix1, gpu_matrix2, gpu_matrix_res)
    print("%s seconds is time for calculation on GPU" % (time.time() - start_time))
    print("End of GPU work!\n")


if __name__ == "__main__":
    cpu_calc()
    gpu_calc()
    x = [0.20007705688476562, 1.621795415878296, 13.972699165344238, 105.11987590789795, 956.7778902053833,
         1989.2391362190247]
    y = [0.15534392356872559, 0.16351268768310547, 0.1880006504058838, 0.2145817470550537, 0.25744752883911133,
         0.2705642986297607]
    z = [100, 200, 400, 800, 1600, 2000]
    plt.plot(x, z)
    plt.title("Время работы CPU")
    plt.xlabel("Размер блоков")
    plt.ylabel("Время работы в секундах")
    plt.grid()
    plt.show()

    plt.plot(z, y)
    plt.title("Время работы GPU")
    plt.xlabel("Размер блоков")
    plt.ylabel("Время работы в секундах")
    plt.grid()
    plt.show()
    print("Конец, спасибо за внимание!\n")
    vsp = []
    for i in range(len(x)):
        vsp.append(x[i]/y[i])

    plt.plot(z,vsp)
    plt.title("Ускорение")
    plt.xlabel("Размер блоков")
    plt.grid()
    plt.show()

# 100 200 400 800 1600 2000
# CPU: 0.20007705688476562,1.621795415878296,13.972699165344238, 105.11987590789795,956.7778902053833,1989.2391362190247
# GPU: 0.24534392356872559,0.18351268768310547,0.2180006504058838,0.1945817470550537,0.20744752883911133, 0.5405642986297607
