import random
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
        self.radius=radius
        self.i_l_outside, self.j_l_outside, self.i_l_inside, self.j_l_inside = self.create_c_inner_c_outer_lists()

        self.count_loop = 0
        self.paused = False
        self.animation = None

    def load_points(self, points_x: list, points_y: list):
        if len(points_x) != len(points_y):
            raise Exception('Lists are not eaqual!')
        for i in range(len(points_x)):
            self.matrix[points_y[i]][points_x[i]] = 1
            self.matrix_2[points_y[i]][points_x[i]] = 1

    def load_file(self, file):
        """Ładuje plik z danymi."""

        lista = []
        with open(file, 'r') as file:
            for line in file:
                lista.append(list(map(lambda e: float(e), line.replace('\n', '').split())))
        for i in range(len(lista)):
            self.matrix[int(lista[i][1])][int(lista[i][0])] = 1

    def create_c_inner_c_outer_lists(self):
        radius = self.radius
        i, j = radius - 1, radius - 1

        i_l_outside = []
        j_l_outside = []
        i_l_inside = []
        j_l_inside = []

        for i_p in range(i - radius, i + radius):
            for j_p in range(j - radius, j + radius):
                if (i_p - i) ** 2 + (j_p - j) ** 2 < radius ** 2:
                    if ((i_p < i - 1) or (i_p > i + 1)) or ((j_p < i - 1) or (j_p > i + 1)):
                        i_l_outside.append(i - i_p)
                        j_l_outside.append(j - j_p)
                    else:
                        i_l_inside.append(i - i_p)
                        j_l_inside.append(j - j_p)

        return i_l_outside, j_l_outside, i_l_inside, j_l_inside


    def count_cells(self, matrix, i, j):
        count_outer = 0
        count_inner = 0
        all_cells_outer = len(self.i_l_outside)
        all_cells_inner = len(self.i_l_inside)

        for el in range(len(self.i_l_outside)):
            try:
                if matrix[i - self.i_l_outside[el]][j - self.j_l_outside[el]] != 0:
                    count_outer += 1
            except:
                pass
        for el in range(len(self.i_l_inside)):
            try:
                if matrix[i - self.i_l_inside[el]][j - self.j_l_inside[el]] == 1:
                    count_inner += 1
            except:
                pass

        u_o = count_outer / all_cells_outer
        u_i = count_inner / all_cells_inner
        return u_o, u_i

    def check_born_or_die(self, i, j):

        if self.switcher:
            matrix = self.matrix
            matrix_2 = self.matrix_2
        else:
            matrix = self.matrix_2
            matrix_2 = self.matrix

        u_o, u_i = self.count_cells(matrix=matrix, i=i, j=j)
        # rules
        if u_i >= 0.5 and 0.26 <= u_o <= 0.46:
            matrix_2[i][j] = 1
        elif u_i < 0.5 and 0.27 <= u_o <= 0.36:
            matrix_2[i][j] = 1
        else:
            matrix_2[i][j] = 0

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
                    self.check_born_or_die(i, j)

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

        self.animation = animation.FuncAnimation(fig, func=game_of_life_loop, frames=200, interval=10, cache_frame_data=False)
        # fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
        # self.animation.save('lenia.gif')
        plt.show()


gra_w_zycie = GameOfLife(n=200, m=200, radius=5, backend='macosx')
# gra_w_zycie.load_file('data.dat')
gra_w_zycie.load_points(points_x=[random.randint(0,199) for _ in range(10000)], points_y=[random.randint(0,199) for _ in range(10000)])
gra_w_zycie.core()

# '1234/12' WOW!
# 12345/12
# 234/345
# 238/3 pulsar
