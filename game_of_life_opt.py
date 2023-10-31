import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.animation import FuncAnimation


class GameOfLife:
    def __init__(self, n, m, radius, rules: str, backend='TkAgg'):
        matplotlib.use(backend)
        self.matrix = np.zeros((n, m))
        self.matrix_2 = np.zeros((n, m))
        self.switcher = True

        self.radius = radius
        self.i_l, self.j_l = self.create_kernel()

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

    def create_kernel(self):
        radius = self.radius
        i, j = radius - 1, radius - 1

        i_l = []
        j_l = []

        for i_p in range(i - radius, i + radius):
            for j_p in range(j - radius, j + radius):
                if (i_p - i) ** 2 + (j_p - j) ** 2 < radius ** 2:
                    if i_p == i and j_p == j:
                        pass
                    else:
                        i_l.append(i - i_p)
                        j_l.append(j - j_p)

        return i_l, j_l

    def count_cells(self, matrix, i, j):
        count = 0

        for el in range(len(self.i_l)):
            try:
                if matrix[i - self.i_l[el]][j - self.j_l[el]] == 1:
                    count += 1
            except:
                pass

        return count

    def check_born_or_die(self, i, j):

        if self.switcher:
            matrix = self.matrix
            matrix_2 = self.matrix_2
        else:
            matrix = self.matrix_2
            matrix_2 = self.matrix

        count = self.count_cells(matrix=matrix, i=i, j=j)

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
            plt.title(f'Rule: {self.rules_str}| Generation: {self.count_loop}| people: {np.count_nonzero(matrix)}')
            self.count_loop += 1
            return im,

        self.animation = FuncAnimation(fig, func=game_of_life_loop, frames=60, interval=10, cache_frame_data=False)
        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
        plt.show()


gra_w_zycie = GameOfLife(n=300, m=300, radius=2, rules='23/3', backend='macosx')
# gra_w_zycie.load_file('data.dat')
# gra_w_zycie.load_points(points_x=[100, 100, 101, 100, 99], points_y=[100, 99, 99, 101, 100])
gra_w_zycie.load_points(points_x=[10, 11, 11, 12, 12], points_y=[10, 11, 12, 11, 10])
gra_w_zycie.core()

# '1234/12' WOW!
# 12345/12
# 234/345
# 238/3 pulsar
