import os
import cv2
import time
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


def load_template_map(image_dim):
    template_path = 'input/template_wafer_map.pkl'
    template = pd.read_pickle(template_path)
    template = cv2.resize(template.waferMap.copy(), dsize=(image_dim[0], image_dim[1]),
                          interpolation=cv2.INTER_NEAREST)

    # 2 - паттерн
    # 1 - фон
    # 0 - область, где нет ничего
    template[template == 2] = 1
    return template


IMAGE_DIMS = (96, 96)
POINT_NUMBER = 300
template_map = load_template_map(IMAGE_DIMS)


class BasisGenerator(object):
    """Класс шаблон для генератора паттерна"""

    # TODO: инициализация пустого паттерна и параметров пластины
    def __init__(self):
        self.basis_points = POINT_NUMBER
        self.wafer_dims = IMAGE_DIMS  # размер пластины
        self.template_map = template_map  # пустой шаблон пластины

    # TODO: вернуть пустой паттерн
    def __call__(self, wafer, *args, **kwargs):
        return self.template_map


class ScratchGenerator(BasisGenerator):
    """Класс для добавления паттерна Царапина на пластину"""
    def __init__(self):
        super(ScratchGenerator, self).__init__()

    # TODO: функционал класса:
    #       1) генерация паттерна на заданном шаблоне с заданными параметрами модели
    #       2) генерация паттерна на чистом шаблоне с незаданными (сгенерировать самому) параметрами модели
    def __call__(self, wafer=None, length=None, all_xc=None, all_yc=None, all_angle=None, line_count=None):
        """

        :param wafer:
        :param length:
        :param xc:
        :param yc:
        :param angle:
        :param line_count:
        :return:
        """

        """ Задать параметры синтеза паттерна, если они не заданы пользавателем """
        # если пластина не задана, применить паттерн к пустому шаблону
        if wafer is None:
            wafer = self.template_map

        if line_count is None:
            # задать количество составных прямых для генерации паттерна
            line_count = np.random.randint(1, 4)

        if all_yc is None:
            # задать стартовую точку для прямой
            yc = [np.random.randint(int(0.3 * self.wafer_dims[0]), int(0.7 * self.wafer_dims[0]))]
            for _ in range(line_count - 1):
                # смещение по y для старта следующей прямой
                delta_yc = np.random.randint(int(0.01 * self.wafer_dims[0]), int(0.02 * self.wafer_dims[0] + 2))
                yc.append(delta_yc)

        if all_xc is None:
            xc = [np.random.randint(int(0.2 * self.wafer_dims[0]), int(0.5 * self.wafer_dims[0]))]
            for _ in range(line_count - 1):
                # смещение по x для старта следующей прямой
                delta_xc = np.random.randint(int(0.01 * self.wafer_dims[0]), int(0.02 * self.wafer_dims[0] + 2))
                np.random.shuffle(delta_xc)
                xc.append(delta_xc)

        if all_angle is None:
            # углы наклона для каждого отрезка
            angle = [np.random.randint(-50, 50) * np.pi / 180]
            for _ in range(line_count - 1):
                part_angle = np.random.randint(30, 40) * np.pi / 180 * np.sign(angle[0])
                angle.append(part_angle)

        if length is None:
            length = np.random.randint(int(0.2 * self.wafer_dims[0]), int(0.45 * self.wafer_dims[0]))

        """ Построить паттерн для пласитны по элементам царапины """
        COLOR_SCALE = 2
        x0, y0 = 0, 0
        for line_iteration in range(line_count):
            # TODO: зачем это?
            # if line_iteration:
            #     step = random.randint(0, size - 1)

            # параметры уравнения
            def delta_(x, y):
                return int(np.sqrt(x ** 2 + y ** 2))

            delta = np.vectorize(delta_)
            cur_length = length - np.sum(delta(all_xc, all_yc)[1:])
            N = 200

            # кусочное построение пилообразной прямой
            # случайное удлинение или укорочение отрезка
            rand = random.randint(-1, 0)
            scale = 0.4
            t = np.linspace(0, length // (line_count + rand * scale), N)

            xc = all_xc[line_iteration]
            yc = all_yc[line_iteration]
            X = np.around(np.cos(all_angle[line_iteration]) * t + xc + x0)
            Y = np.around(np.sin(all_angle[line_iteration]) * t + yc + y0)

            x_prev, y_prev = x0, y0
            x_first, y_first = 0, 0
            for j in range(X.shape[0]):
                x = int(X[j])
                y = int(Y[j])
                if j == 0:
                    # первая точка прямой
                    x_first, y_first = x, y
                try:
                    if wafer[x, y] == 1:
                        wafer[x, y] = COLOR_SCALE
                        x0, y0 = x, y
                except IndexError:
                    break

            # сшивка прямых
            if line_iteration != 0:
                # уравнение прямой сшивки
                k = (y_prev - y_first) / (x_prev - x_first + 1e-06)
                b = y_first - k * x_first
                X = np.linspace(x_prev, x_first, 20)
                Y = k * X + b
                X_ = np.around(X)
                Y_ = np.around(Y)
                for j in range(X_.shape[0]):
                    x = int(X_[j])
                    y = int(Y_[j])
                    try:
                        if wafer[x, y] == 1:
                            wafer[x, y] = COLOR_SCALE
                    except IndexError:
                        break


class PoisonGenerator(BasisGenerator):
    """Класс для добавления Пуассоновского точечного процесса на пластину"""
    def __init__(self):
        super(PoisonGenerator, self).__init__()

    # TODO: заложить два функционала:
    #       1) случайный локальный процесс вдоль заданной маски паттерна
    #       2) случайны локальный процесс на всей пластине
    def __call__(self, wafer, *args, **kwargs):
        pass


class MorphNoiseGenerator(BasisGenerator):
    """Класс для добавления шумов на пластину"""
    def __init__(self):
        super(MorphNoiseGenerator, self).__init__()

    # TODO: перенести функцию добавления шума с помощью морфологических операций
    def __call__(self, wafer, *args, **kwargs):
        pass


if __name__ == '__main__':
    pass
