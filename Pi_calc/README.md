# Лабораторная работая  "Pi_Monte-Carlo"
## Задача: 
  Реализовать алгоритм вычисления числа PI. Язык: C++ или Python.
    
**Входные данные**: количество точек.
     
**Выходные данные**: время выполнения и полученные числа PI.
     
**Характеристики системы**: 

Процессор: 11th Gen Intel(R) Core(TM) i7-11700F @ 2.50GHz (16 CPUs).

Видеокарта: NVIDIA GeForce RTX 3070.

Реализация выполнена на языке **Python**.

В данной лабораторной работе параллельные вычисления значения числа Пи (π) на центральном процессоре (CPU) и графическом процессоре (GPU) с использованием библиотеки Numba и ее подмодуля CUDA для работы с CUDA-ядрами

Измерения проводились при количество итераций для вычисления π = 32. Код программы представлен в файле. Количество нитей на **GPU = 32**.

Эксперимент проводится на случайно сгенерированных данных  **в области (0, 0) - (1, 1)**, и результаты времени выполнения сравниваются между CPU и GPU для разного числа точек N.
Задачи распараллеливание на CUDA:


В таблице представлены результаты:
|Номер итерации| Время работы GPU в секундах | Время работы CPU в секундах | Пи на GPU|Пи на CPU |Ускорение GPU относительно CPU|  Кол-во точек|
|--------|-------------|:----------------:|-----------------:|--------------:|-------------:|---------------:|
|1|       0.830821 |     0.017952 |  3.155200 |  3.136888 |  0.021607   |      10000 |
|2|       0.004986 |     0.038896 |  3.157800 |  3.140962 |  7.800564   |      20000 |
|3|       0.005985 |    0.056847  | 3.150800  | 3.139792  | 9.498646    |      30000 |
|4|       0.006981 |     0.074800 |  3.147100 |  3.143997 | 10.714559   |      40000 |
|5|       0.008973 |     0.095772 |  3.143040 |  3.141920 | 10.672955   |      50000 |
|6|       0.010970 |     0.112698 |  3.147067 |  3.141048 | 10.272966   |      60000 |
|7|       0.012965 |     0.124666 |  3.147543 |  3.141398 |  9.615465   |      70000 |
|8|       0.013967 |     0.147634 |  3.147450 |  3.141100 | 10.570321   |      80000 |
|9|       0.016987 |     0.178522 |  3.145378 |  3.142837 | 10.509151   |      90000 |
|10|      0.017952 |     0.195444 |  3.143720 |  3.141142 | 10.887204   |     100000 |




На рисунке предствален график ускорения работы  программы на GPU по сравнению с CPU : 

![image](https://github.com/andiyash/high-performance-computing/assets/145579445/a96b0d14-3d8f-4690-b7e0-d2459c376ea0)




Результаты указывают на то, что при увеличении размерности  ускорение работы на GPU увеличивается, при этом важно, что на маленькой размерности CPU производит вычисление быстрее чем GPU. Это может быть обусловлено тем, что для создания большого количества потоков расходуются дополнительные ресурсы. Максимального ускорения **10.887204**  удалось достичь при  количестве точек = 100000. 