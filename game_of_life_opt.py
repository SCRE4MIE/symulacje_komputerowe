import random

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation
import time
import numba


class GameOfLife:
    def __init__(self, n, m, kernel_outer_radius, kernel_inner_radius, rules: str, backend='TkAgg'):
        matplotlib.use(backend)
        self.matrix = np.zeros((n, m))
        self.matrix_2 = np.zeros((n, m))
        self.switcher = True
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
                if (0 <= i_matrix_index < self.n) and (0 <= j_matrix_index < self.m):
                    count += matrix[i_matrix_index][j_matrix_index] * self.kernel[i_k][j_k]

        return count

    def check_born_or_die(self, i, j):

        if self.switcher:
            matrix = self.matrix
            matrix_2 = self.matrix_2
        else:
            matrix = self.matrix_2
            matrix_2 = self.matrix

        count = self.count_cells(matrix=matrix, i_c=i, j_c=j)

        if matrix[i][j] == 0:  # born
            if count in self.rules_born:
                matrix_2[i][j] = 1
            else:
                matrix_2[i][j] = 0

        if matrix[i][j] == 1:  # die
            if count not in self.rules_die:
                matrix_2[i][j] = 0
            else:
                matrix_2[i][j] = 1

    def toggle_pause(self, *args, **kwargs):
        if self.paused:
            self.animation.resume()
        else:
            self.animation.pause()
        self.paused = not self.paused

    def core(self):

        fig= plt.figure(figsize=(10, 8))
        im = plt.imshow(self.matrix, cmap='gray', animated=True)

        def game_of_life_loop(frame):
            start = time.process_time()
            for i in range(self.matrix.shape[0]):
                for j in range(self.matrix.shape[1]):
                    self.check_born_or_die(i, j)

            if self.switcher:
                self.switcher = False
                matrix = self.matrix
            else:
                self.switcher = True
                matrix = self.matrix_2
            print(time.process_time() - start)
            im.set_array(matrix)
            plt.title(f'Rule: {self.rules_str}| Generation: {self.count_loop}| people: {np.count_nonzero(matrix)}')
            self.count_loop += 1
            print(self.count_loop)
            return im,

        self.animation = FuncAnimation(fig, func=game_of_life_loop, frames=30, interval=1, cache_frame_data=False)
        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
        # plt.show()
        self.animation.save('gof.gif')


gra_w_zycie = GameOfLife(n=600, m=600, kernel_inner_radius=1, kernel_outer_radius=10, rules='23/3', backend='macosx')
gra_w_zycie.load_file('data.dat')
gra_w_zycie.load_points(points_x=[100, 100, 101, 100, 99], points_y=[100, 99, 99, 101, 100])
# gra_w_zycie.load_points(points_x=[10, 11, 11, 12, 12], points_y=[10, 11, 12, 11, 10])
gra_w_zycie.load_points(points_x=[random.randint(0, 599) for _ in range(1000)],
                points_y=[random.randint(0, 599) for _ in range(1000)])
gra_w_zycie.core()

# '1234/12' WOW!
# 12345/12
# 234/345
# 238/3 pulsar
