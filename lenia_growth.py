import math
import random

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.animation as animation


class GameOfLife:
    def __init__(self, n, m, radius,  backend='TkAgg'):
        matplotlib.use(backend)
        self.matrix = np.zeros((n, m))
        self.matrix_2 = np.zeros((n, m))
        self.switcher = True
        self.radius = radius
        self.i_l, self.j_l = self.create_kernel()

        self.count_loop = 0
        self.paused = False
        self.animation = None

    def load_points(self, points_x: list, points_y: list):
        if len(points_x) != len(points_y):
            raise Exception('Lists are not eaqual!')
        for i in range(len(points_x)):
            self.matrix[points_y[i]][points_x[i]] = 1
            self.matrix_2[points_y[i]][points_x[i]] = 1

    def load_points_random(self, points_x: list, points_y: list):
        if len(points_x) != len(points_y):
            raise Exception('Lists are not eaqual!')
        for i in range(len(points_x)):
            self.matrix[points_y[i]][points_x[i]] = random.uniform(0, 1)

    def load_file(self, file):
        """Ładuje plik z danymi."""

        lista = []
        with open(file, 'r') as file:
            for line in file:
                lista.append(list(map(lambda e: float(e), line.replace('\n', '').split())))
        for i in range(len(lista)):
            self.matrix[int(lista[i][1])][int(lista[i][0])] = 1

    def create_kernel(self):
        radius = self.radius
        i, j = radius - 1, radius - 1

        i_l = []
        j_l = []

        for i_p in range(i - radius, i + radius):
            for j_p in range(j - radius, j + radius):
                if radius / 2 < (i_p - i) ** 2 + (j_p - j) ** 2 < radius ** 2:
                    i_l.append(i - i_p)
                    j_l.append(j - j_p)

        return i_l, j_l

    def calc_U(self, matrix, i, j):
        count_outer = 0
        all_cells_outer = len(self.i_l)

        for el in range(len(self.i_l)):
            try:
                count_outer += matrix[i - self.i_l[el]][j - self.j_l[el]]
            except:
                pass
        u = count_outer / all_cells_outer
        return u

    def growth_func(self, u):
        sigma = 0.138
        mu = 0
        l = abs(u - mu)
        k = 2 * (sigma ** 2)
        return 2 * np.exp(-(l ** 2) / k) - 1

    def calc_cell_value(self, i, j):

        if self.switcher:
            matrix = self.matrix
            matrix_2 = self.matrix_2
        else:
            matrix = self.matrix_2
            matrix_2 = self.matrix

        u = self.calc_U(matrix=matrix, i=i, j=j)
        a = self.growth_func(u)
        matrix_2[i][j] = min(max((matrix[i][j] + 0.1*a), 0), 1)


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
            for i in range(self.matrix.shape[0]):
                for j in range(self.matrix.shape[1]):
                    self.calc_cell_value(i, j)

            if self.switcher:
                self.switcher = False
                matrix = self.matrix
            else:
                self.switcher = True
                matrix = self.matrix_2

            im.set_array(matrix)
            plt.title(f'Generation: {self.count_loop}| people: {np.count_nonzero(matrix)}')
            self.count_loop += 1
            return im,

        self.animation = animation.FuncAnimation(fig, func=game_of_life_loop, frames=50, interval=10, cache_frame_data=False)
        # fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
        # self.animation.save('lenia.gif')
        plt.show()


gra_w_zycie = GameOfLife(n=200, m=200, radius=3, backend='macosx')
gra_w_zycie.load_file('data.dat')
# gra_w_zycie.load_points_random(points_x=[random.randint(0,199) for _ in range(10000)], points_y=[random.randint(0,199) for _ in range(10000)])
# gra_w_zycie.load_points(points_x=[10, 11, 11, 12, 12], points_y=[10, 11, 12, 11, 10])
gra_w_zycie.core()

# '1234/12' WOW!
# 12345/12
# 234/345
# 238/3 pulsar
