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

IMAGE_DIMS = (96, 96)
PATTERN_COLOR = 3  # цвет паттерна (стандартный)
DEFECT_COLOR = 2  # цвет всех дефектов
WAFER_COLOR = 1  # цвет пластины без дефектов
BACK_COLOR = 0  # цвет пустого поля


def load_template_map(image_dim):
    template_path = '../../input/template_wafer_map.pkl'
    template = pd.read_pickle(template_path)
    template = cv2.resize(template.waferMap.copy(), dsize=(image_dim[0], image_dim[1]),
                          interpolation=cv2.INTER_NEAREST)

    # 2 - паттерн
    # 1 - фон
    # 0 - область, где нет ничего
    template[template == DEFECT_COLOR] = WAFER_COLOR
    return template


def create_zero_template_map(image_dim):
    """
    Создать пустой шаблон пластины
    :param image_dim: tuple: размер пласитны в пикселях
    :return: numpy.ndarray: шаблон пластины
    """
    x = np.arange(0, image_dim[0])
    y = np.arange(0, image_dim[1])
    arr = np.zeros((y.size, x.size))
    cx = image_dim[0] // 2
    cy = image_dim[1] // 2
    r = image_dim[0] // 2

    mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
    arr[mask] = WAFER_COLOR

    return arr.astype(np.uint8)


class BasisGenerator(object):
    """ Класс шаблон для генератора паттерна """

    def __init__(self):
        self.wafer_dims = IMAGE_DIMS  # размер пластины
        self.template_map = create_zero_template_map(IMAGE_DIMS)  # пустой шаблон пластины
        self.pattern_color = PATTERN_COLOR  # цвет паттерна
        self.defect_color = DEFECT_COLOR  # цвет всех дефектов
        self.wafer_color = WAFER_COLOR  # цвет пластины без дефектов
        self.back_color = BACK_COLOR  # цвет пустого поля

    def __call__(self, wafer=None, mask=None, *args, **kwargs):
        if wafer is None:
            return self.template_map
        else:
            return wafer

    def pattern_regularization(self, wafer, pattern_mask, lam_poisson):
        """
        Регуляризаци дефекта с помощью пуассоновсокго точечного процесса вдоль маски дефекта
        :param wafer: np.ndarray: пластина для нанесения паттерна
        :param pattern_mask: np.ndarray: маска дефекта
        :param lam_poisson: float: величина лямбды в распределении Пуассона
        :return: np.ndarray, np.ndarray: пластина с паттерном и маска паттерна
        """
        # сформировать пуассоновкие точки от 0 до 1
        random_poisson = np.random.poisson(lam=lam_poisson, size=pattern_mask.shape)
        random_poisson = random_poisson / np.amax(random_poisson)

        # наложить на объект и очистить малые значения
        pattern_mask = pattern_mask + random_poisson
        pattern_mask[pattern_mask <= self.pattern_color] = self.back_color
        pattern_mask[pattern_mask > self.pattern_color] = self.pattern_color - self.wafer_color

        # вырезать оригинальный паттерн и наложить новый "шумный"
        wafer[wafer == self.pattern_color] = self.wafer_color
        wafer = wafer + pattern_mask

        return wafer, pattern_mask


class ScratchGenerator(BasisGenerator):
    """ Класс для добавления паттерна "Scratch" на пластину """

    def __init__(self):
        super(ScratchGenerator, self).__init__()
        self.pattern_color = 10  # цвет паттерна Scratch

    def __call__(self, wafer=None, mask=None, is_noise=False, lam_poisson=1.2,
                 length=None, line_weight=None, all_xc=None, all_yc=None, all_angle=None, part_line_count=None):
        """
        Сгенерировать паттерн "Scratch" на заданной пластине.
        :param wafer: np.ndarray: пластина для нанесения паттерна
        :param mask: np.ndarray: маска дефекта
        :param is_noise: bool: если True добавить регуляризацию на паттерн, False - ничего не делать
        :param lam_poisson: float: величина лямбды(частота событий) в распределении Пуассона
        :param length: int: длина паттерна "Scratch"
        :param line_weight: int: толщина паттерна "Scratch"
        :param all_xc: list: список точек (по x) старта для каждой составной части прямой
        :param all_yc: list: список точек (по y) старта для каждой составной части прямой
        :param all_angle: list: список навправлений для старта каждой составной части прямой
        :param part_line_count: int: количество составных частей прямой
        :return: np.ndarray, np.ndarray: пластина с паттерном и маска паттерна
        """

        """ Задать параметры синтеза паттерна, если они не заданы пользавателем """
        # если пластина не задана, применить паттерн к пустому шаблону
        if wafer is None:
            wafer = deepcopy(self.template_map)

        if part_line_count is None:
            # задать количество составных прямых для генерации паттерна
            part_line_count = np.random.randint(2, 5)

        if all_yc is None:
            # задать стартовую точку для прямой
            all_yc = [np.random.randint(int(0.3 * self.wafer_dims[0]), int(0.7 * self.wafer_dims[0]))]
            for _ in range(part_line_count - 1):
                # смещение по y для старта следующей прямой
                delta_yc = np.random.randint(int(0.01 * self.wafer_dims[0]), int(0.02 * self.wafer_dims[0] + 2))
                all_yc.append(delta_yc)

        if all_xc is None:
            all_xc = [np.random.randint(int(0.2 * self.wafer_dims[0]), int(0.5 * self.wafer_dims[0]))]
            for _ in range(part_line_count - 1):
                # смещение по x для старта следующей прямой
                delta_xc = np.random.randint(int(0.01 * self.wafer_dims[0]), int(0.02 * self.wafer_dims[0] + 2))
                all_xc.append(delta_xc)

        if all_angle is None:
            # углы наклона для каждого отрезка
            all_angle = [np.random.randint(-50, 50) * np.pi / 180]
            for _ in range(part_line_count - 1):
                part_angle = np.random.randint(30, 40) * np.pi / 180 * np.sign(all_angle[0])
                all_angle.append(part_angle)

        if length is None:
            length = np.random.randint(int(0.2 * self.wafer_dims[0]), int(0.45 * self.wafer_dims[0]))

        if line_weight is None:
            # если толщина царапины не задана — применить толщину по умолчанию в 1 пиксель
            line_weight = 5.0

        # длина всех составных частей прямой без учета линий соединения частей
        part_length = length - int(np.sum(np.linalg.norm([all_xc[1:], all_yc[1:]])))
        x_part_zero, y_part_zero = 0, 0  # кооридинаты начала составной части прямой
        N = 100  # количество точек в прямой

        """ Построить паттерн для пласитны по элементам царапины """
        for line_iteration in range(part_line_count):
            """ Параметризация уравнения составной части прямой """
            # длина составной линии отрезка - случайное удлинение или сокращение составной длины отрезка
            rand, scale = random.randint(-1, 0), 0.4
            t = np.linspace(0, part_length // (part_line_count + rand * scale), N)

            # текущая точка старта составной части прямой
            xc, yc = all_xc[line_iteration], all_yc[line_iteration]
            # сгенерировать все точки составной части прямой
            _X_idx = np.around(np.cos(all_angle[line_iteration]) * t + xc + x_part_zero).astype(int)
            _Y_idx = np.around(np.sin(all_angle[line_iteration]) * t + yc + y_part_zero).astype(int)

            x_prev, y_prev = x_part_zero, y_part_zero  # предыдущие точки старта прямой
            x_first, y_first = None, None  # текущие точки старта новой прямой
            for _x, _y in zip(_X_idx, _Y_idx):  # цикл по всем точкам составной прямой
                if x_first is None and y_first is None:
                    x_first, y_first = _x, _y  # первая точка составной прямой
                try:
                    if wafer[_x, _y] == self.wafer_color:  # расположение место пластины
                        low_weight = int(np.floor(line_weight / 2.0))  # большая часть толщины линии
                        high_weight = int(np.ceil(line_weight / 2.0))  # меньшая часть толщины линии
                        temp_window = wafer[_x - low_weight: _x + high_weight, _y - low_weight:_y + high_weight]
                        # окно для отрисовки участка линии (отдельная переменная для повышения читабельности кода)
                        temp_window = np.where(temp_window == self.wafer_color, self.pattern_color, temp_window)
                        # отрисовываем часть линии, если мы на пластине
                        wafer[_x - low_weight: _x + high_weight, _y - low_weight:_y + high_weight] = temp_window
                        del temp_window
                        x_part_zero, y_part_zero = _x, _y
                except IndexError:  # закончить построение при выходе за границы пластины
                    break

            """ Сшить составные прямые """
            if line_iteration != 0:
                # задать параметры  уравнение прямой сшивки
                k = (y_prev - y_first) / (x_prev - x_first + 1e-06)
                b = y_first - k * x_first
                # сгенерировать все уравнения сшивки
                _X_idx = np.around(np.linspace(x_prev, x_first, 20)).astype(int)
                _Y_idx = np.around(k * _X_idx + b).astype(int)
                for _x, _y in zip(_X_idx, _Y_idx):  # цикл по всем точкам уравнения сшивки
                    try:
                        if wafer[_x, _y] == self.wafer_color:  # расположение место пластины
                            wafer[_x, _y] = self.pattern_color
                    except IndexError:  # закончить построение при выходе за границы пластины
                        break

        # выделить маску паттерна
        pattern_mask = deepcopy(wafer)
        pattern_mask[pattern_mask != self.pattern_color] = self.back_color

        # регуляризация дефекта с помощью шума
        if is_noise:
            wafer, pattern_mask = self.pattern_regularization(wafer, pattern_mask, lam_poisson)

        # добавить в маске слой для нового цвета
        if mask is None:
            mask = np.expand_dims(pattern_mask, axis=2)
        else:
            mask = np.dstack((mask, pattern_mask))

        wafer[wafer == self.pattern_color] = self.defect_color  # нормировка значений

        return wafer, mask


class RingGenerator(BasisGenerator):
    """ Класс для формирования колец для добавления паттернов "Donut", "Edge-Ring", "Center" etc на пластину """

    def __init__(self):
        super(RingGenerator, self).__init__()
        self.pattern_color = 11

    def donut_params(self, x_center=None, y_center=None, density=None,
                     sector_angle_start=None, sector_angle_end=None, radius_inner=None, radius_outer=None):
        """
        Параметры для генерация паттерна "Donut"
        :param x_center: int: центр кольца по X
        :param y_center: int: центр кольца по Y
        :param density: int: плотность дефектов
        :param sector_angle_start: float: начало сектора кольца
        :param sector_angle_end: float: начало сектора кольца
        :param radius_inner: int: внутренний радиус кольца
        :param radius_outer: int: наружный радиус кольца
        :return: все параметры
        """

        rand_sector = np.random.randint(0, 4)
        self.pattern_color = 12

        if sector_angle_start is None:
            sector_angle_start = np.random.uniform(95 * rand_sector, 30 + 95 * rand_sector) * np.pi / 180
        if sector_angle_end is None:
            sector_angle_end = np.random.uniform(180 + 90 * rand_sector, 360 * (rand_sector + 1)) * np.pi / 180

        if x_center is None:
            x_center = np.random.randint(0.45 * self.wafer_dims[0], 0.55 * self.wafer_dims[0])
        if y_center is None:
            y_center = np.random.randint(0.45 * self.wafer_dims[0], 0.55 * self.wafer_dims[0])

        if density is None:
            density = np.random.randint(200, 210)

        if radius_inner is None:
            radius_inner = np.random.randint(0.15 * self.wafer_dims[0], 0.3 * self.wafer_dims[0])
        if radius_outer is None:
            radius_outer = np.random.randint(0.33 * self.wafer_dims[0], 0.4 * self.wafer_dims[0])

        return x_center, y_center, sector_angle_start, sector_angle_end, density, radius_inner, radius_outer

    def loc_params(self, x_center=None, y_center=None, density=None,
                   sector_angle_start=None, sector_angle_end=None, radius_inner=None, radius_outer=None):
        """
        Параметры для генерация паттерна "Loc" TODO поработать на улучшением репрезантативности(после консультации)
        :param x_center: int: центр кольца по X
        :param y_center: int: центр кольца по Y
        :param density: int: плотность дефектов
        :param sector_angle_start: float: начало сектора кольца
        :param sector_angle_end: float: начало сектора кольца
        :param radius_inner: int: внутренний радиус кольца
        :param radius_outer: int: наружный радиус кольца
        :return: все параметры
        """

        rand_sector = np.random.randint(0, 4)
        self.pattern_color = 13

        if sector_angle_start is None:
            sector_angle_start = np.random.uniform(95 * rand_sector, 55 + 90 * rand_sector) * np.pi / 180
        if sector_angle_end is None:
            sector_angle_end = np.random.uniform(65 + 90 * rand_sector, 95 * (rand_sector + 1)) * np.pi / 180

        if x_center is None:
            x_center = np.random.randint(0.45 * self.wafer_dims[0], 0.55 * self.wafer_dims[0])
        if y_center is None:
            y_center = np.random.randint(0.45 * self.wafer_dims[0], 0.55 * self.wafer_dims[0])

        if density is None:
            density = np.random.randint(200, 210)

        if radius_inner is None:
            radius_inner = np.random.randint(0.1 * self.wafer_dims[0], 0.2 * self.wafer_dims[0])
        if radius_outer is None:
            radius_outer = np.random.randint(0.2 * self.wafer_dims[0], 0.25 * self.wafer_dims[0])

        return x_center, y_center, sector_angle_start, sector_angle_end, density, radius_inner, radius_outer

    def center_params(self, x_center=None, y_center=None, density=None,
                      sector_angle_start=None, sector_angle_end=None, radius_inner=None, radius_outer=None):
        """
        Параметры для генерация паттерна "Center"
        :param x_center: int: центр кольца по X
        :param y_center: int: центр кольца по Y
        :param density: int: плотность дефектов
        :param sector_angle_start: float: начало сектора кольца
        :param sector_angle_end: float: начало сектора кольца
        :param radius_inner: int: внутренний радиус кольца
        :param radius_outer: int: наружный радиус кольца
        :return: все параметры
        """

        rand_sector = np.random.randint(0, 4)
        self.pattern_color = 14

        if sector_angle_start is None:
            sector_angle_start = np.random.uniform(95 * rand_sector, 10 + 95 * rand_sector) * np.pi / 180
        if sector_angle_end is None:
            sector_angle_end = np.random.uniform(45 + 90 * rand_sector, 95 * (rand_sector + 1)) * np.pi / 180

        if x_center is None:
            x_center = np.random.randint(0.48 * self.wafer_dims[0], 0.50 * self.wafer_dims[0])
        if y_center is None:
            y_center = np.random.randint(0.48 * self.wafer_dims[0], 0.50 * self.wafer_dims[0])

        if density is None:
            density = np.random.randint(200, 210)

        if radius_inner is None:
            radius_inner = np.random.randint(0.0 * self.wafer_dims[0], 0.05 * self.wafer_dims[0])
        if radius_outer is None:
            radius_outer = np.random.randint(0.12 * self.wafer_dims[0], 0.23 * self.wafer_dims[0])

        return x_center, y_center, sector_angle_start, sector_angle_end, density, radius_inner, radius_outer

    def edge_ring_params(self, x_center=None, y_center=None, density=None,
                         sector_angle_start=None, sector_angle_end=None, radius_inner=None, radius_outer=None):
        """
        Параметры для генерация паттерна "Edge-ring"
        :param x_center: int: центр кольца по X
        :param y_center: int: центр кольца по Y
        :param density: int: плотность дефектов
        :param sector_angle_start: float: начало сектора кольца
        :param sector_angle_end: float: начало сектора кольца
        :param radius_inner: int: внутренний радиус кольца
        :param radius_outer: int: наружный радиус кольца
        :return: все параметры
        """

        rand_sector = np.random.randint(0, 4)
        self.pattern_color = 15
        center = 0.5 * self.wafer_dims[0]

        if sector_angle_start is None:
            sector_angle_start = np.random.uniform(90 * rand_sector, 30 + 90 * rand_sector) * np.pi / 180
        if sector_angle_end is None:
            sector_angle_end = np.random.uniform(320 + 90 * rand_sector, 360 * (rand_sector + 1)) * np.pi / 180

        if x_center is None:
            x_center = np.random.randint(center - 2, center)
        if y_center is None:
            y_center = np.random.randint(center - 2, center)

        if density is None:
            density = np.random.randint(200, 210)

        if radius_inner is None:
            radius_inner = np.random.randint(center - 4, center - 3)
        if radius_outer is None:
            radius_outer = np.random.randint(center, center + 1)

        return x_center, y_center, sector_angle_start, sector_angle_end, density, radius_inner, radius_outer

    def edge_loc_params(self, x_center=None, y_center=None, density=None,
                        sector_angle_start=None, sector_angle_end=None, radius_inner=None, radius_outer=None):
        """
        Параметры для генерация паттерна "Edge-loc"
        :param x_center: int: центр кольца по X
        :param y_center: int: центр кольца по Y
        :param density: int: плотность дефектов
        :param sector_angle_start: float: начало сектора кольца
        :param sector_angle_end: float: начало сектора кольца
        :param radius_inner: int: внутренний радиус кольца
        :param radius_outer: int: наружный радиус кольца
        :return: все параметры
        """

        rand_sector = np.random.randint(0, 4)
        self.pattern_color = 16
        center = 0.5 * self.wafer_dims[0]

        if sector_angle_start is None:
            sector_angle_start = np.random.uniform(15 + 90 * rand_sector, 25 + 90 * rand_sector) * np.pi / 180
        if sector_angle_end is None:
            sector_angle_end = np.random.uniform(55 + 90 * rand_sector, 115 + 90 * (rand_sector + 1)) * np.pi / 180

        if x_center is None:
            x_center = np.random.randint(center - 2, center + 1)
        if y_center is None:
            y_center = np.random.randint(center - 2, center + 1)

        if density is None:
            density = np.random.randint(200, 210)

        if radius_inner is None:
            radius_inner = np.random.randint(center - 5, center - 3)
        if radius_outer is None:
            radius_outer = np.random.randint(center, center + 1)

        return x_center, y_center, sector_angle_start, sector_angle_end, density, radius_inner, radius_outer

    def __call__(self, wafer=None, mask=None, is_noise=False, lam_poisson=1.5, pattern_type=None):
        """
        Сгенерировать один из паттернов "Donut", "Edge-Ring", "Loc", "Edge-Loc", "Center" на заданной пластине
        :param wafer: np.ndarray: пластина для нанесения паттерна
        :param mask: np.ndarray: маска дефекта
        :param is_noise: bool: если True добавить регуляризацию на паттерн, False - ничего не делать
        :param lam_poisson: float: величина лямбды(частота событий) в распределении Пуассона
        :param pattern_type: str: класс паттерна ("Donut", "Edge-Ring", "Loc", "Edge-Loc", "Center")
        :return: np.ndarray, np.ndarray: пластина с паттерном и маска паттерна
        """

        """ Задать параметры синтеза колец, если они не заданы пользавателем """
        # если пластина не задана, применить паттерн к пустому шаблону
        if wafer is None:
            wafer = deepcopy(self.template_map)

        pattern_params = {  # TODO Внести по умолчанию или ошибку Undefined Pattern
            'Donut': self.donut_params(),
            'Loc': self.loc_params(),
            'Center': self.center_params(),
            'Edge-Ring': self.edge_loc_params(),
            'Edge-Loc': self.edge_loc_params()
        }

        ######################## TODO Прокомментировать алгоритм
        # данные в зависимости от паттерна
        x_center, y_center, sector_angle_start, sector_angle_end, density, radius_inner, radius_outer = \
            pattern_params[pattern_type]

        # параметры кольца, объединяем
        sector_angle = np.linspace(sector_angle_start, sector_angle_end, density)
        radius_ring = np.linspace(radius_inner, radius_outer, density)

        # синтез сетки
        radius_ring_, sector_angle_ = np.meshgrid(radius_ring, sector_angle)
        x_array = radius_ring_ * (np.cos(sector_angle_)) + x_center
        y_array = radius_ring_ * (np.sin(sector_angle_)) + y_center
        x_array_ = np.around(x_array)
        y_array_ = np.around(y_array)

        # индексы для полигона
        points = []
        for x_ in range(x_array_.shape[0]):
            for y_ in range(x_array_.shape[1]):
                x = x_array_[x_, y_]
                y = y_array_[x_, y_]
                points.append((x, y))

        for idx in points:
            x_, y_ = idx
            x_ = int(round(x_))
            y_ = int(round(y_))
            try:
                if wafer[x_, y_] == self.wafer_color:
                    wafer[x_, y_] = self.pattern_color
            except IndexError:
                break

        # выделить маску паттерна
        pattern_mask = deepcopy(wafer)
        pattern_mask[pattern_mask != self.pattern_color] = self.back_color

        # регуляризация дефекта с помощью шума
        if is_noise:
            wafer, pattern_mask = self.pattern_regularization(wafer, pattern_mask, lam_poisson)

        # добавить в маске слой для нового цвета
        if mask is None:
            mask = np.expand_dims(pattern_mask, axis=2)
        else:
            mask = np.dstack((mask, pattern_mask))

        wafer[wafer == self.pattern_color] = self.defect_color  # нормировка значений

        return wafer, mask


###########################


class NoiseGenerator(BasisGenerator):
    """ Класс для добавления шумов на пластину """

    def __init__(self):
        super(NoiseGenerator, self).__init__()

    def __call__(self, wafer=None, mask=None, lam_poisson=0.3, **kwargs):
        """
        Внести шум на пластину
        :param wafer: np.ndarray: пластина для нанесения паттерна
        :param mask: np.ndarray: маска дефекта
        :param lam_poisson: float: величина лямбды в распределении Пуассона
        :return: np.ndarray, np.ndarray: пластина с паттерном и маска паттерна
        """
        noise_wafer, mask = self.noise_poisson(wafer, mask, lam_poisson=lam_poisson)

        return noise_wafer, mask

    def noise_poisson(self, wafer, mask, lam_poisson=0.3):
        """
        Внести случайный шум на карту с помощью пуассоновсокого точечного процесса
        :param wafer: np.ndarray: пластина для нанесения паттерна
        :param mask: np.ndarray: маска дефекта
        :param lam_poisson: float: величина лямбды в распределении Пуассона
        :return: np.ndarray, np.ndarray: пластина с паттерном и маска паттерна
        """

        # TODO: заложить дополнительный функционал:
        #       1) случайны локальный процесс на всей пластине

        # внесем шум
        noise_img = deepcopy(wafer)
        # подготовить маску с местоположением шума на карте
        noise_mask = np.random.randint(0, 2, size=noise_img.shape).astype(np.bool)
        noise_mask[noise_img == 0] = False
        # случайное распределение шума
        r = np.random.poisson(lam=lam_poisson, size=noise_img.shape)

        r[r == self.back_color] = self.wafer_color  # убрать значения с области фона
        r[r > self.defect_color] = self.defect_color  # нормировать все значения на цвет дефекта
        noise_img[noise_mask] = r[noise_mask]  # применить шум

        # вернуть дефект с каждого слоя маски
        for layer in range(mask.shape[2]):
            noise_img[mask[:, :, layer] > self.back_color] = self.defect_color

        return noise_img, mask


def mask_for_visualize(input_mask):
    """
    Форматировать маску с паттернами, записанными по слоям, в маску со всеми паттернами сразу для построения
    :param input_mask: np.ndarray (w, h, dims): маска с паттернами по слоям
    :return: np.ndarray (w, h): маска со всеми паттернами на одном слое
    """
    h, w, dims = input_mask.shape
    new_mask = np.zeros((h, w))

    for dim in range(dims):  # пройтись по каждому слою
        color = np.max(input_mask[:, :, dim])  # найти цвет паттерна в слое
        new_mask += input_mask[:, :, dim] / color  # отнормирвать текущий паттерн на свой цвет
    new_mask /= dims  # отнормировать на все паттерны

    return new_mask


if __name__ == '__main__':
    """ Проверка функций """
    scratch_generator = ScratchGenerator()  # тестовый генератор
    morph_generator = NoiseGenerator()
    ring_generator = RingGenerator()

    example_count = 5  # примеров для тестирования
    for i in range(example_count):
        fig, maxs = plt.subplots(1, 2, figsize=(10, 7))

        # сгенерировать несколько паттернов на одной пластине
        wafer, mask = None, None
        pattern_count = 1  # количество паттернов на пластине
        for i in range(pattern_count):
            # wafer, mask = scratch_generator(wafer, mask, line_weight=1, is_noise=True, lam_poisson=1.7)
            wafer, mask = ring_generator(wafer, mask, pattern_type="Donut", is_noise=True)
            wafer, mask = morph_generator(wafer, mask)

        # отрисовать результат
        maxs[0].imshow(wafer, cmap='inferno')
        maxs[1].imshow(mask_for_visualize(mask), cmap='inferno')
        plt.show()

    # plt.imshow(create_zero_template_map(IMAGE_DIMS))
    # plt.show()
