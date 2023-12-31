
# Лабораторная работая  "Mass_search"
## Задача: 
- Реализовать mass_search Язык: C++ или Python.
    - Входные данные: матрицы строк с различными размерностями.
    - Выходные данные: проверка корректности  + время вычисления.
    - Характеристики системы: 

Процессор: 11th Gen Intel(R) Core(TM) i7-11700F @ 2.50GHz (16 CPUs).

Видеокарта: NVIDIA GeForce RTX 3070.

Реализация выполнена на языке **Python**.
В данной лабораторной работе производился запуск программы для сложения двух векторов с использованием технологии CUDA. В программе необходимо выполнить поиск подстрок в матрицах с различными разрешениями. 

Измерения проводились на размерностях GridDim (8,8) и BlockDim(16,16). Код программы представлен в файле. Использовалась библиотека numba.  В таблицах будут представлены усредненые значения по 12 запускам.<br />

Эксперимент проводится на случайно сгенерированных данных (N и H), и результаты времени выполнения сравниваются между CPU и GPU для разных размеров данных.
Задачи распараллеливание на CUDA:
Данные копируются с центрального процессора на графический процессор и обратно для выполнения вычислений.  

В таблице представлены результаты:
| Разрешение  | Время работы CPU в секундах | Время работы GPU в секундах|Ускорение GPU относительно CPU|
|-------------|:----------------:|-----------------:|---------------:|
| 160  | 0.00518506  | 0.02782562 | 0.18634108 | 
| 320  | 0.01027582   | 0.00089405 | 11.49361316 | 
| 480  | 0.01586111  | 0.00099685 | 15.91119562 | 
| 640  | 0.02154551 | 0.00089729 |24.01179753 |
| 800 | 0.02632918  | 0.00099728 | 26.40096584 | 
| 960 | 0.03221061	 | 0.0008986 | 35.84534359 | 
| 1120 | 0.03680146	 | 0.00099757 | 36.89120719 |
| 1280 | 0.04189363	 | 0.00109999 | 38.08539784 | 
| 1440 | 0.04696512	 | 0.0011004 | 42.68015773 | 
| 1600 | 0.05335381	 | 0.00089746 | 59.45010892  |




На рисунке предствален график ускорения работы  программы на GPU по сравнению с CPU : 

![image](https://github.com/andiyash/high-performance-computing/assets/145579445/7ba90c69-167e-405a-a82d-dc08372d366a)




Результаты указывают на то, что при увеличении размерности  ускорение работы на GPU увеличивается, при этом важно, что на маленькой размерности CPU производит поиск быстрее чем GPU. Это может быть обусловлено тем, что для создания большого количества потоков расходуются лишние ресурсы. Максимального ускорения 59.45010892 удалось достичь при  GridDim (2,2),BlockDim(16,16) при размерности 1600. 
