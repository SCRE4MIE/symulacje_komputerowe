import copy
import threading

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from joblib import Parallel, cpu_count, delayed
from matplotlib.animation import FuncAnimation
import time
import multiprocessing


class GameOfLife:
    def __init__(self, n, m, kernel_outer_radius, kernel_inner_radius, rules: str, backend='TkAgg'):
        matplotlib.use(backend)
        # self.matrix = np.zeros((n, m))
        # self.matrix_tmp = np.zeros((n, m))
        self.matrix = np.frombuffer(multiprocessing.Array('i', n * m).get_obj(), dtype=np.int32).reshape(n, m)
        self.matrix_tmp = np.frombuffer(multiprocessing.Array('i', n * m).get_obj(), dtype=np.int32).reshape(n, m)
        self.n = n
        self.m = m

        self.kernel = self.create_kernel(kernel_outer_radius, kernel_inner_radius)
        self.shape_kernel = len(self.kernel), len(self.kernel[0])
        self.half_size_i_kernel = int(self.shape_kernel[0] / 2)
        self.half_size_j_kernel = int(self.shape_kernel[1] / 2)

        self.rules_str = rules
        self.rules_born, self.rules_die = self.translate_rules()
        self.count_loop = 0
        self.paused = False
        self.animation = None

    def translate_rules(self):
        split_rules_str = self.rules_str.split('/')
        rules_to_born = [int(c) for c in split_rules_str[1]]
        rules_to_die = [int(c) for c in split_rules_str[0]]
        return rules_to_born, rules_to_die


    def load_points(self, points_x: list, points_y: list):
        if len(points_x) != len(points_y):
            raise Exception('Lists are not eaqual!')
        for i in range(len(points_x)):
            self.matrix[points_y[i]][points_x[i]] = 1

    def load_file(self, file):
        """Ładuje plik z danymi."""

        lista = []
        with open(file, 'r') as file:
            for line in file:
                lista.append(list(map(lambda e: float(e), line.replace('\n', '').split())))
        for i in range(len(lista)):
            self.matrix[int(lista[i][1])][int(lista[i][0])] = 1

    def create_kernel(self, outer_radius, inner_radius):
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

    def count_cells(self, matrix, i_c, j_c):
        count = 0

        for i_k in range(self.shape_kernel[0]):
            for j_k in range(self.shape_kernel[1]):
                i_matrix_index = i_c - self.half_size_i_kernel + i_k
                j_matrix_index = j_c - self.half_size_j_kernel + j_k
                # if (0 <= i_matrix_index < self.n) and (0 <= j_matrix_index < self.m):
                #     count += matrix[i_matrix_index][j_matrix_index] * self.kernel[i_k][j_k]
                count += matrix[i_matrix_index % self.n][j_matrix_index % self.m] * self.kernel[i_k][j_k]

        return count

    def check_born_or_die(self, i, j):

        count = self.count_cells(matrix=self.matrix, i_c=i, j_c=j)

        if self.matrix[i][j] == 0:  # born
            if count in self.rules_born:
                self.matrix_tmp[i][j] = 1
            else:
                self.matrix_tmp[i][j] = 0

        if self.matrix[i][j] == 1:  # die
            if count not in self.rules_die:
                self.matrix_tmp[i][j] = 0
            else:
                self.matrix_tmp[i][j] = 1

    def task(self, start_i, end_i, start_j, end_j):
        for i_o in range(start_i, end_i):
            for j_o in range(start_j, end_j):
                self.check_born_or_die(i_o, j_o)


    def core(self, frame):




        start = time.process_time()

        p1 = multiprocessing.Process(target=self.task, args=(0, int(self.n / 2), 0, int(self.m / 2)))
        p2 = multiprocessing.Process(target=self.task, args=(0, int(self.n / 2), int(self.m / 2), self.m))
        p3 = multiprocessing.Process(target=self.task, args=(int(self.n / 2), self.n, 0, int(self.m / 2)))
        p4 = multiprocessing.Process(target=self.task, args=(int(self.n / 2), self.n, int(self.m / 2), self.m))
        p1.start()
        p2.start()
        p3.start()
        p4.start()
        # p1.join()
        # p2.join()
        # p3.join()
        # p4.join()

        # self.task(0, int(self.n / 2), 0, int(self.m / 2))
        # self.task(0, int(self.n / 2), int(self.m / 2), self.m)
        # self.task(int(self.n / 2), self.n, 0, int(self.m / 2))
        # self.task(int(self.n / 2), self.n, int(self.m / 2), self.m)

        print(time.process_time() - start)
        plt.matshow(self.matrix_tmp)
        plt.show()
        self.matrix = copy.deepcopy(self.matrix_tmp)
        self.matrix_tmp = np.zeros((self.n, self.m))
        im.set_array(self.matrix)
        plt.title(f'Rule: {self.rules_str}| Generation: {self.count_loop}| people: {np.count_nonzero(self.matrix)}')
        self.count_loop += 1
        return im,







if __name__ == '__main__':
    gra_w_zycie = GameOfLife(n=200, m=200, kernel_inner_radius=1, kernel_outer_radius=2, rules='23/3', backend='macosx')
    gra_w_zycie.load_file('data.dat')
    gra_w_zycie.load_points(points_x=[100, 100, 101, 100, 99], points_y=[100, 99, 99, 101, 100])
    # gra_w_zycie.load_points(points_x=[10, 11, 11, 12, 12], points_y=[10, 11, 12, 11, 10])
    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(gra_w_zycie.matrix, cmap='gray', animated=True)
    anim=FuncAnimation(fig, func=gra_w_zycie.core, frames=60, interval=10, cache_frame_data=False)
    plt.show()

