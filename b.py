from multiprocessing import Process

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation
import time
import multiprocessing as mp
from IPython.display import clear_output
import concurrent.futures
import multiprocessing
import threading

matrix = None
matrix_2 = None
switcher = True
iters = None
n = None
m = None

kernel = None
shape_kernel = None

rules_str = None
rules_born, rules_die = None, None
count_loop = 0


def initial_values(n_m, m_m, it, kernel_outer_radius, kernel_inner_radius, rules: str, backend='TkAgg'):
    global switcher
    switcher = True
    global iters
    iters = it
    global n
    n = n_m
    global m
    m = m_m

    global matrix
    matrix = np.zeros((n, m))
    global matrix_2
    matrix_2 = np.zeros((n, m))

    global kernel
    kernel = create_kernel(kernel_outer_radius, kernel_inner_radius)
    global shape_kernel
    shape_kernel = len(kernel), len(kernel[0])

    global rules_str
    rules_str = rules

    global rules_born, rules_die
    rules_born, rules_die = translate_rules()
    global count_loop
    count_loop = 0


def translate_rules():
    split_rules_str = rules_str.split('/')
    rules_to_born = [int(c) for c in split_rules_str[1]]
    rules_to_die = [int(c) for c in split_rules_str[0]]
    return rules_to_born, rules_to_die


def load_points(points_x: list, points_y: list):
    if len(points_x) != len(points_y):
        raise Exception('Lists are not eaqual!')
    for i in range(len(points_x)):
        matrix[points_y[i]][points_x[i]] = 1


def load_file(file):
    """Ładuje plik z danymi."""

    lista = []
    with open(file, 'r') as file:
        for line in file:
            lista.append(list(map(lambda e: float(e), line.replace('\n', '').split())))
    for i in range(len(lista)):
        matrix[int(lista[i][1])][int(lista[i][0])] = 1


def create_kernel(outer_radius, inner_radius):
    size = 2 * outer_radius - 1
    kernel = np.zeros((size, size))
    center = outer_radius - 1

    for i in range(size):
        for j in range(size):
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
            if distance < inner_radius:
                kernel[i][j] = 0
            elif distance < outer_radius:
                kernel[i][j] = 1

    return kernel


def count_cells(matrix, i_c, j_c):
    count = 0
    for i_k in range(shape_kernel[0]):
        for j_k in range(shape_kernel[1]):
            i_matrix_index = i_c - int(shape_kernel[0] / 2) + i_k
            j_matrix_index = j_c - int(shape_kernel[1] / 2) + j_k
            if (0 <= i_matrix_index < len(matrix)) and (0 <= j_matrix_index < len(matrix[0])):
                count += matrix[i_matrix_index][j_matrix_index] * kernel[i_k][j_k]

    return count


def check_born_or_die(i, j):
    if switcher:
        matrix_tmp = matrix
        matrix_2_tmp = matrix_2
    else:
        matrix_tmp = matrix_2
        matrix_2_tmp = matrix

    count = count_cells(matrix=matrix_tmp, i_c=i, j_c=j)

    if matrix_tmp[i][j] == 0:  # born
        if count in rules_born:
            matrix_2_tmp[i][j] = 1
        else:
            matrix_2_tmp[i][j] = 0

    if matrix_tmp[i][j] == 1:  # die
        if count not in rules_die:
            matrix_2_tmp[i][j] = 0
        else:
            matrix_2_tmp[i][j] = 1


def task(start_i, end_i, start_j, end_j):
    for i_o in range(start_i, end_i):
        for j_o in range(start_j, end_j):
            check_born_or_die(i_o, j_o)


def core():
    global switcher
    tasks = [
        (0, int(n / 2), 0, int(m / 2)),
        (0, int(n / 2), int(m / 2), m),
        (int(n / 2), n, 0, int(m / 2)),
        (int(n / 2), n, int(m / 2), m),
    ]
    procs = []
    proc = Process(target=task, args=tasks[0])
    procs.append(proc)
    proc.start()
    proc = Process(target=task, args=tasks[1])
    procs.append(proc)
    proc.start()
    proc = Process(target=task, args=tasks[2])
    procs.append(proc)
    proc.start()
    proc = Process(target=task, args=tasks[0])
    procs.append(proc)
    proc.start()

    if switcher:
        switcher = False
        matrix_tmp = matrix
    else:
        switcher = True
        matrix_tmp = matrix_2

    plt.figure(figsize=(10, 10))
    plt.matshow(matrix_tmp, cmap='Greys', fignum=1)
    plt.title(f'Generation, people: {np.count_nonzero(matrix)}')
    plt.show()


if __name__ == '__main__':
    initial_values(n_m=200, m_m=200, it=1000, kernel_inner_radius=1, kernel_outer_radius=2, rules='23/3',
                   backend='macosx')
    load_file('data.dat')
    for i in range(1):
        core()