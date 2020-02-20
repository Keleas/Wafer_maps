import cv2
import pandas as pd
import numpy as np
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import functools
from sklearn.datasets import make_blobs

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

    wafer_mask = (x[np.newaxis, :] - cx) ** 2 + (y[:, np.newaxis] - cy) ** 2 < r ** 2
    arr[wafer_mask] = WAFER_COLOR

    return arr.astype(np.uint8)


def retry_generator(retry_limit, minimal_size=None):
    """
    Декоратор для повторного вызова генератора дефектов.
    :param retry_limit: int: количество попыток повтора
    :param minimal_size: int: минимальный размер дефекта паттерна
    :return: decorator: результат удовлятворяющий условию
    """
    if minimal_size is None:
        minimal_size = 1
        # TODO: Разобраться как её связать с классом для задания минимального размера от размера пластины

    def decorator(function_to_decorate):
        @functools.wraps(function_to_decorate)
        def wrapper(*args, **kwargs):
            for _ in range(retry_limit):
                wafer, pattern_mask = function_to_decorate(*args, **kwargs)
                if np.count_nonzero(pattern_mask) >= minimal_size:
                    return wafer, pattern_mask

        return wrapper

    return decorator


class BasisGenerator(object):
    """ Класс шаблон для генератора паттерна """

    def __init__(self):
        self.wafer_dims = IMAGE_DIMS  # размер пластины
        self.template_map = create_zero_template_map(self.wafer_dims)  # пустой шаблон пластины
        self.pattern_color = PATTERN_COLOR  # цвет паттерна
        self.defect_color = DEFECT_COLOR  # цвет всех дефектов
        self.wafer_color = WAFER_COLOR  # цвет пластины без дефектов
        self.back_color = BACK_COLOR  # цвет пустого поля

    def __call__(self, wafer=None, pattern_mask_array=None, *args, **kwargs):
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
        :return: wafer, pattern_mask: np.ndarray, np.ndarray: пластина с паттерном и маска паттерна
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

    @staticmethod
    def mask_stacking(pattern_mask_array, new_mask):
        """
        Добавление новой полученной маски паттерна к уже существующим
        :param pattern_mask_array: nd.array: маски паттернов данной пластины
        :param new_mask: nd.array: новая маска паттерна дефекта
        :return: pattern_mask_array: nd.array: обновленные маски паттернов данной пластины
        """
        if pattern_mask_array is None:
            # если массив пустой, то есть новая маска является первой в ней, формируем массив из только неё
            pattern_mask_array = np.expand_dims(new_mask, axis=0)
        else:
            # в противном случае изменяем её размерность и объединяем
            new_mask = np.expand_dims(new_mask, axis=0)
            pattern_mask_array = np.concatenate((pattern_mask_array, new_mask), axis=0)
        return pattern_mask_array


class ScratchGenerator(BasisGenerator):
    """ Класс для добавления паттерна "Scratch" на пластину """

    def __init__(self):
        super(ScratchGenerator, self).__init__()
        self.pattern_color = 10  # цвет паттерна Scratch

    def __call__(self, wafer=None, pattern_mask_array=None, is_noise=False, lam_poisson=1.2,
                 length=None, line_weight=None, all_xc=None, all_yc=None, all_angle=None, part_line_count=None):
        """
        Сгенерировать паттерн "Scratch" на заданной пластине.
        :param wafer: np.ndarray: пластина для нанесения паттерна
        :param pattern_mask_array: np.ndarray: маска дефекта
        :param is_noise: bool: если True добавить регуляризацию на паттерн, False - ничего не делать
        :param lam_poisson: float: величина лямбды(частота событий) в распределении Пуассона
        :param length: int: длина паттерна "Scratch"
        :param line_weight: int: толщина паттерна "Scratch"
        :param all_xc: list: список точек (по x) старта для каждой составной части прямой
        :param all_yc: list: список точек (по y) старта для каждой составной части прямой
        :param all_angle: list: список навправлений для старта каждой составной части прямой
        :param part_line_count: int: количество составных частей прямой
        :return: wafer, pattern_mask_array: np.ndarray, np.ndarray: пластина с паттерном и маска паттерна
        """

        """ Задать параметры синтеза паттерна, если они не заданы пользавателем """
        # если пластина не задана, применить паттерн к пустому шаблону
        if wafer is None:
            wafer = deepcopy(create_zero_template_map(self.wafer_dims))

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
            line_weight = 1.0

        # длина всех составных частей прямой без учета линий соединения частей
        part_length = length - int(np.sum(np.linalg.norm([all_xc[1:], all_yc[1:]])))
        x_part_zero, y_part_zero = 0, 0  # кооридинаты начала составной части прямой
        density = 100  # количество точек в прямой

        """ Построить паттерн для пласитны по элементам царапины """
        for line_iteration in range(part_line_count):
            """ Параметризация уравнения составной части прямой """
            # длина составной линии отрезка - случайное удлинение или сокращение составной длины отрезка
            rand, scale = random.randint(-1, 0), 0.4
            t = np.linspace(0, part_length // (part_line_count + rand * scale), density)

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
                        _temp_window = wafer[_x - low_weight: _x + high_weight, _y - low_weight:_y + high_weight]
                        # окно для отрисовки участка линии (отдельная переменная для повышения читабельности кода)
                        _temp_window = np.where(_temp_window == self.wafer_color, self.pattern_color, _temp_window)
                        # отрисовываем часть линии, если мы на пластине
                        wafer[_x - low_weight: _x + high_weight, _y - low_weight:_y + high_weight] = _temp_window
                        del _temp_window
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
        pattern_mask_array = self.mask_stacking(pattern_mask_array, pattern_mask)

        wafer[wafer == self.pattern_color] = self.defect_color  # нормировка значений

        return wafer, pattern_mask_array


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
        """

        rand_sector = np.random.randint(0, 4)
        self.pattern_color = 12

        if sector_angle_start is None:
            sector_angle_start = np.random.uniform(95 * rand_sector, 30 + 95 * rand_sector) * np.pi / 180
        if sector_angle_end is None:
            sector_angle_end = np.random.uniform(180 + 90 * rand_sector, 360 * (rand_sector + 1)) * np.pi / 180

        if x_center is None:
            x_center = np.random.randint(int(0.45 * self.wafer_dims[0]), int(0.55 * self.wafer_dims[0]))
        if y_center is None:
            y_center = np.random.randint(int(0.45 * self.wafer_dims[0]), int(0.55 * self.wafer_dims[0]))

        if density is None:
            density = np.random.randint(200, 210)

        if radius_inner is None:
            radius_inner = np.random.randint(int(0.15 * self.wafer_dims[0]), int(0.3 * self.wafer_dims[0]))
        if radius_outer is None:
            radius_outer = np.random.randint(int(0.33 * self.wafer_dims[0]), int(0.4 * self.wafer_dims[0]))

        return x_center, y_center, sector_angle_start, sector_angle_end, density, radius_inner, radius_outer

    def loc_params(self, x_center=None, y_center=None, density=None,
                   sector_angle_start=None, sector_angle_end=None, radius_inner=None, radius_outer=None):
        """
        Параметры для генерация паттерна "Loc" TODO: поработать на улучшением репрезантативности(после консультации)
        :param x_center: int: центр кольца по X
        :param y_center: int: центр кольца по Y
        :param density: int: плотность дефектов
        :param sector_angle_start: float: начало сектора кольца
        :param sector_angle_end: float: начало сектора кольца
        :param radius_inner: int: внутренний радиус кольца
        :param radius_outer: int: наружный радиус кольца
        :return: x_center, y_center, density, sector_angle_start, sector_angle_end, radius_inner, radius_outer:
        int, int, int, float, float, int ,int: все параметры
        """

        rand_sector = np.random.randint(0, 4)
        self.pattern_color = 13

        if sector_angle_start is None:
            sector_angle_start = np.random.uniform(95 * rand_sector, 55 + 90 * rand_sector) * np.pi / 180
        if sector_angle_end is None:
            sector_angle_end = np.random.uniform(65 + 90 * rand_sector, 95 * (rand_sector + 1)) * np.pi / 180

        if x_center is None:
            x_center = np.random.randint(int(0.45 * self.wafer_dims[0]), int(0.55 * self.wafer_dims[0]))
        if y_center is None:
            y_center = np.random.randint(int(0.45 * self.wafer_dims[0]), int(0.55 * self.wafer_dims[0]))

        if density is None:
            density = np.random.randint(200, 210)

        if radius_inner is None:
            radius_inner = np.random.randint(int(0.1 * self.wafer_dims[0]), int(0.2 * self.wafer_dims[0]))
        if radius_outer is None:
            radius_outer = np.random.randint(int(0.2 * self.wafer_dims[0]), int(0.25 * self.wafer_dims[0]))

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
        :return: x_center, y_center, density, sector_angle_start, sector_angle_end, radius_inner, radius_outer:
        int, int, int, float, float, int ,int: все параметры
        """

        rand_sector = np.random.randint(0, 4)
        self.pattern_color = 14

        if sector_angle_start is None:
            sector_angle_start = np.random.uniform(95 * rand_sector, 10 + 95 * rand_sector) * np.pi / 180
        if sector_angle_end is None:
            sector_angle_end = np.random.uniform(45 + 90 * rand_sector, 95 * (rand_sector + 1)) * np.pi / 180

        if x_center is None:
            x_center = np.random.randint(int(0.48 * self.wafer_dims[0]), int(0.50 * self.wafer_dims[0]))
        if y_center is None:
            y_center = np.random.randint(int(0.48 * self.wafer_dims[0]), int(0.50 * self.wafer_dims[0]))

        if density is None:
            density = np.random.randint(200, 210)

        if radius_inner is None:
            radius_inner = np.random.randint(int(0.0 * self.wafer_dims[0]), int(0.05 * self.wafer_dims[0]))
        if radius_outer is None:
            radius_outer = np.random.randint(int(0.12 * self.wafer_dims[0]), int(0.23 * self.wafer_dims[0]))

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
        :return: x_center, y_center, density, sector_angle_start, sector_angle_end, radius_inner, radius_outer:
        int, int, int, float, float, int ,int: все параметры
        """

        rand_sector = np.random.randint(0, 4)
        self.pattern_color = 15
        center = int(0.5 * self.wafer_dims[0])

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
        :return: x_center, y_center, density, sector_angle_start, sector_angle_end, radius_inner, radius_outer:
        int, int, int, float, float, int ,int: все параметры
        """

        rand_sector = np.random.randint(0, 4)
        self.pattern_color = 16
        center = int(0.5 * self.wafer_dims[0])

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

    def __call__(self, wafer=None, pattern_mask_array=None, lam_poisson=1.5, pattern_type=None):
        """
        Сгенерировать один из паттернов "Donut", "Edge-Ring", "Loc", "Edge-Loc", "Center" на заданной пластине
        :param wafer: np.ndarray: пластина для нанесения паттерна
        :param pattern_mask_array: np.ndarray: маска дефекта
        :param lam_poisson: float: величина лямбды(частота событий) в распределении Пуассона
        :param pattern_type: str: класс паттерна ("Donut", "Edge-Ring", "Loc", "Edge-Loc", "Center")
        :return: wafer, pattern_mask_array: np.ndarray, np.ndarray: пластина с паттерном и маска паттерна
        """

        """ Задать параметры синтеза колец, если они не заданы пользавателем """
        # если пластина не задана, применить паттерн к пустому шаблону
        if wafer is None:
            wafer = deepcopy(create_zero_template_map(self.wafer_dims))

        pattern_params = {
            'Donut': self.donut_params(),
            'Loc': self.loc_params(),
            'Center': self.center_params(),
            'Edge-Ring': self.edge_loc_params(),
            'Edge-Loc': self.edge_loc_params()
        }
        if pattern_type not in pattern_params:
            raise KeyError('Неправильный класс паттерна, возможные варианты: '
                           '("Donut", "Edge-Ring", "Loc", "Edge-Loc", "Center")')

        # данные колец в зависимости от паттерна
        x_center, y_center, sector_angle_start, sector_angle_end, density, radius_inner, radius_outer = \
            pattern_params[pattern_type]

        # параметры кольца, объединение
        sector_angle = np.linspace(sector_angle_start, sector_angle_end, density)
        radius_ring = np.linspace(radius_inner, radius_outer, density)

        # синтез сетки для построения паттерна
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

        # отрисовка паттерна
        for idx in points:
            x_, y_ = idx
            x_ = int(round(x_))
            y_ = int(round(y_))
            try:
                if wafer[x_, y_] == self.wafer_color:  # проверка на наличие пластины/отсутсвие дефекта перед отрисовкой
                    wafer[x_, y_] = self.pattern_color
            except IndexError:
                break

        # выделить маску паттерна
        pattern_mask = deepcopy(wafer)
        pattern_mask[pattern_mask != self.pattern_color] = self.back_color

        # регуляризация дефекта с помощью шума (в данных паттернах — обязательна)
        wafer, pattern_mask = self.pattern_regularization(wafer, pattern_mask, lam_poisson)

        # добавить в маске слой для нового цвета
        pattern_mask_array = self.mask_stacking(pattern_mask_array, pattern_mask)

        wafer[wafer == self.pattern_color] = self.defect_color  # нормировка значений

        return wafer, pattern_mask_array


class ClusterGenerator(BasisGenerator):
    """ Класс для добавления паттернов "Cluster", "Grid" на пластину"""
    def __init__(self):
        super(ClusterGenerator, self).__init__()

    def __call__(self, wafer=None, pattern_mask_array=None, cluster_origin_point=None, cluster_bounding_box=None,
                 density=None, number_of_cluster_centers=None, size_factor=None):
        """
        Сгенерировать паттерн "Cluster" на заданной пластине при помощи изотропного Гауссовского распределения
        :param wafer: np.ndarray: пластина для нанесения паттерна
        :param pattern_mask_array: np.ndarray: маска дефекта
        :param cluster_origin_point: int, int: центр кластера
        :param cluster_bounding_box: int, int: границы размещения центров кластера
        :param density: int: плотность дефектов (количество точек, возможно накладывающихся)
        :param number_of_cluster_centers: int: количество центров кластера
        :param size_factor: float: множитель размеров
        :return: wafer, pattern_mask_array: np.ndarray, np.ndarray: пластина с паттерном и маска патттерна
        """
        # TODO: мб заменить sklearn просто Гауссовским распределениями и т.п.

        # если пластина не задана, применить паттерн к пустому шаблону
        if wafer is None:
            wafer = deepcopy(create_zero_template_map(self.wafer_dims))

        if size_factor is None:
            size_factor = 1.0

        if density is None:
            density = int((size_factor ** 2) * self.wafer_dims[0] ** 2 / 20)

        if cluster_origin_point is None:
            cluster_origin_point = np.random.randint(int(0.45 * self.wafer_dims[0]), int(0.55 * self.wafer_dims[0]))

        if cluster_bounding_box is None:
            bounding = np.random.randint(size_factor * self.wafer_dims[0] * 0.05, size_factor * self.wafer_dims[0] * 0.1)
            cluster_bounding_box = (-bounding, bounding)

        if number_of_cluster_centers is None:
            number_of_cluster_centers = 5 * int(size_factor * self.wafer_dims[0] * 0.1)

        # формируем кластеры в ограниченной зоне
        cluster_coords, _ = make_blobs(n_samples=density, centers=number_of_cluster_centers, n_features=2,
                                       center_box=cluster_bounding_box)

        cluster_coords += cluster_origin_point  # смещаем в соответствии с указанным центром

        cluster_coords = cluster_coords.astype(int)  # переводим в int для перевода в координаты

        pattern_mask = np.zeros(wafer.shape)  # шаблон для маски

        for _x, _y in cluster_coords:
            pattern_mask[_x, _y] = wafer[_x, _y]  # заполняем маску по координатам кластера

        # построение маски паттерна только там, где присутствует сама пластина
        pattern_mask = np.where(pattern_mask == self.wafer_color, self.pattern_color, pattern_mask)

        # добавить в маске слой для нового цвета
        pattern_mask_array = self.mask_stacking(pattern_mask_array, pattern_mask)

        return wafer, pattern_mask_array


class CurvedScratchGenerator(BasisGenerator):
    """ Класс для добавления паттерна "Curved Scratch" на пластину"""

    def __init__(self):
        super(CurvedScratchGenerator, self).__init__()

    @retry_generator(retry_limit=10, minimal_size=1)
    def __call__(self, wafer=None, pattern_mask_array=None, scratch_ring_center=None, scratch_ring_radius=None,
                 scratch_ring_angle=None, line_weight=None, is_noise=False, lam_poisson=0.5):
        """
        Сгенерировать паттерн "Curved Scratch" на заданной пластине.
        :param wafer: np.ndarray: пластина для нанесения паттерна
        :param pattern_mask_array: np.ndarray: маска дефекта
        :param scratch_ring_center: int, int: центр кольца, используемый при построении царапины
        :param scratch_ring_radius: int: радиус кольца, используемый при построении царапины
        :param scratch_ring_angle: int, int : углы дуги царапины
        :param line_weight: int: примерная толщина паттерна "Curved Scratch"
        :param is_noise: bool: если True добавить регуляризацию на паттерн, False - ничего не делать
        :param lam_poisson: float: величина лямбды(частота событий) в распределении Пуассона
        :return: wafer, pattern_mask_array: np.ndarray, np.ndarray: пластина с паттерном и маска паттерна
        """

        if wafer is None:
            wafer = deepcopy(create_zero_template_map(self.wafer_dims))

        if scratch_ring_radius is None:
            # минимальный радиус — четверть пластины, максимальный — половины
            scratch_ring_radius = np.random.randint(int(self.wafer_dims[0] / 4), int(self.wafer_dims[0] / 2))

        if scratch_ring_center is None:
            scratch_ring_center = (np.random.randint(int(-scratch_ring_radius / 2),
                                                     int(self.wafer_dims[0] + scratch_ring_radius / 2)),
                                   np.random.randint(int(-scratch_ring_radius / 2),
                                                     int(self.wafer_dims[0] + scratch_ring_radius / 2)))

        if scratch_ring_angle is None:
            # минимальный угол — 10 градусов

            # condition_out_of_box_x = (scratch_ring_center[0] < 0) or (scratch_ring_center[0] > self.wafer_dims[0])
            # condition_out_of_box_y = (scratch_ring_center[1] < 0) or (scratch_ring_center[1] > self.wafer_dims[1])
            #
            # if condition_out_of_box_x or condition_out_of_box_y:
            #     starting_angle =
            # else:
            starting_angle = np.random.randint(0, 180)
            scratch_ring_angle = (starting_angle, np.random.randint(starting_angle + 10, starting_angle + 359))

        if line_weight is None:
            # если толщина царапины не задана — применить толщину по умолчанию в 1 пиксель
            line_weight = 1.0

        self.pattern_color = 22

        starting_angle = np.radians(-scratch_ring_angle[0])  # начало угла сектора в радианах
        ending_angle = np.radians(-scratch_ring_angle[1])  # конец угла сектора в радианах

        _x, _y = scratch_ring_center[0], scratch_ring_center[1]  # разделяем координаты центра на x и y координату

        _x1 = _x + scratch_ring_radius * np.cos(starting_angle)  # точка x первой прямой для построения дуги
        _y1 = _x + scratch_ring_radius * np.sin(starting_angle)  # точка y первой прямой для построения дуги

        _x2 = _x + scratch_ring_radius * np.cos(ending_angle)  # точка x второй прямой для построения дуги
        _y2 = _x + scratch_ring_radius * np.sin(ending_angle)  # точка y второй прямой для построения дуги

        pattern_mask = np.zeros(wafer.shape)  # шаблон для маски

        x = np.arange(0, pattern_mask.shape[0])  # массив х координат маски
        y = np.arange(0, pattern_mask.shape[1])  # массив y координат маски

        # построение маски при помощи пересечения областей/внутри снаружи круга для построения кольца
        scratch_coord_inner = (x[np.newaxis, :] - _x) ** 2 + (y[:, np.newaxis] - _y) ** 2 \
                              <= scratch_ring_radius ** 2 + scratch_ring_radius * line_weight

        scratch_coord_outer = (x[np.newaxis, :] - _x) ** 2 + (y[:, np.newaxis] - _y) ** 2 \
                              >= scratch_ring_radius ** 2 - scratch_ring_radius * line_weight

        # построение маски при помощи параметрических уравнений вспомогательных прямых для ограничения дуги
        scratch_coord_sector_1 = \
            x[np.newaxis, :] * (_y - _y1) - y[:, np.newaxis] * (_x - _x1) <= -(_x - _x1) * _x + (_y - _y1) * _y
        scratch_coord_sector_2 = \
            x[np.newaxis, :] * (_y - _y2) - y[:, np.newaxis] * (_x - _x2) >= -(_x - _x2) * _x + (_y - _y2) * _y

        # объединение масок
        scratch_coord_ring = np.logical_and(scratch_coord_inner, scratch_coord_outer)
        scratch_coord_sector = np.logical_and(scratch_coord_sector_1, scratch_coord_sector_2)

        if (scratch_ring_angle[1] - scratch_ring_angle[0]) <= 180:
            scratch_coords = np.logical_and(scratch_coord_ring, scratch_coord_sector)
        else:
            scratch_coords = np.logical_and(scratch_coord_ring, ~scratch_coord_sector)

        print(scratch_coords)

        pattern_mask[scratch_coords] = wafer[scratch_coords]  # копирование вафли из данного окна в маску

        # построение маски паттерна только там, где присутствует сама пластина
        pattern_mask = np.where(pattern_mask == self.wafer_color, self.pattern_color, pattern_mask)

        # регуляризация дефекта с помощью шума
        if is_noise:
            wafer, pattern_mask = self.pattern_regularization(wafer, pattern_mask, lam_poisson)

        # добавить в маске слой для нового цвета
        pattern_mask_array = self.mask_stacking(pattern_mask_array, pattern_mask)

        return wafer, pattern_mask_array


class LocGenerator(BasisGenerator):
    """ Класс для добавления локализованного пуассоновского шума на пластину"""

    def __init__(self):
        super(LocGenerator, self).__init__()

    def __call__(self, wafer=None, pattern_mask_array=None, window_location=None, window_size=None,
                 shape=None, lam_poisson=0.3, additional_size=None):
        """
        Регуляризаци дефекта с помощью пуассоновсокго точечного процесса вдоль маски дефекта
        :param wafer: np.ndarray: пластина для нанесения паттерна
        :param window_location: int, int: координата центра окна
        :param window_size: int: размер окна (радиус в случае "Circle", полуось в случае "Ellipse")
        :param shape: str: форма окна
        :param lam_poisson: float: величина лямбды в распределении Пуассона
        :param additional_size: int: дополнительный размер, используемый для построения эллипса/прямоугольника
        :return: wafer, pattern_mask_array: np.ndarray, np.ndarray: пластина с паттерном и маска паттерна
        """

        shape_dict = {'Circle', 'Square', 'Ellipse', 'Rectangle'}  # словарь всех форм окна
        if shape not in shape_dict:
            raise KeyError('Неправильный тип окна. Возможные варианты: '
                           '"Square", "Circle", "Ellipse", "Rectangle"')
        if wafer is None:
            wafer = deepcopy(self.template_map)

        if window_location is None:
            window_location = [np.random.randint(0, wafer.shape[0]), np.random.randint(0, wafer.shape[1])]

        if window_size is None:
            window_size = np.random.randint(5, int(self.wafer_dims[0] / 4))

        if additional_size is None:
            additional_size = np.random.randint(5, window_size + 1)

        self.pattern_color = 21
        _x, _y = window_location[0], window_location[1]

        loc_mask = np.zeros(wafer.shape)

        if shape is None or "Square":
            x = np.arange(0, loc_mask.shape[0])
            y = np.arange(0, loc_mask.shape[1])

            loc_window = np.abs((x[np.newaxis, :] - _x) + (y[:, np.newaxis] - _y)) \
                         + np.abs((x[np.newaxis, :] - _x) - (y[:, np.newaxis] - _y)) < 2 * window_size

            loc_mask[loc_window] = wafer[loc_window]  # копирование вафли из данного окна в маску
            loc_mask = np.where(loc_mask == self.wafer_color, self.pattern_color, loc_mask)

        if shape is "Rectangle":
            x = np.arange(0, loc_mask.shape[0])
            y = np.arange(0, loc_mask.shape[1])

            loc_window = np.abs((x[np.newaxis, :] - _x) * additional_size + (y[:, np.newaxis] - _y) * window_size) \
                         + np.abs((x[np.newaxis, :] - _x) * additional_size - (y[:, np.newaxis] - _y) * window_size) \
                         < 2 * window_size * additional_size

            loc_mask[loc_window] = wafer[loc_window]  # копирование вафли из данного окна в маску
            loc_mask = np.where(loc_mask == self.wafer_color, self.pattern_color, loc_mask)

        if shape is "Circle":
            x = np.arange(0, loc_mask.shape[0])
            y = np.arange(0, loc_mask.shape[1])

            loc_window = (x[np.newaxis, :] - _x) ** 2 + (y[:, np.newaxis] - _y) ** 2 < window_size ** 2

            loc_mask[loc_window] = wafer[loc_window]  # копирование вафли из данного окна в маску
            loc_mask = np.where(loc_mask == self.wafer_color, self.pattern_color, loc_mask)

        if shape is "Ellipse":
            x = np.arange(0, loc_mask.shape[0])
            y = np.arange(0, loc_mask.shape[1])

            loc_window = (additional_size ** 2) * (x[np.newaxis, :] - _x) ** 2 \
                         + (window_size ** 2) * (y[:, np.newaxis] - _y) ** 2 < (window_size * additional_size) ** 2

            loc_mask[loc_window] = wafer[loc_window]  # копирование вафли из данного окна в маску
            loc_mask = np.where(loc_mask == self.wafer_color, self.pattern_color, loc_mask)

        wafer, pattern_mask = self.pattern_regularization(wafer, loc_mask, lam_poisson)  # пуассоновский шум по маске
        wafer[wafer == self.pattern_color] = self.defect_color  # нормировка значений

        # добавить в маске слой для нового цвета
        pattern_mask_array = self.mask_stacking(pattern_mask_array, pattern_mask)

        return wafer, pattern_mask_array


class NoiseGenerator(BasisGenerator):
    """ Класс для добавления шумов на пластину """

    def __init__(self):
        super(NoiseGenerator, self).__init__()

    def __call__(self, wafer=None, pattern_mask_array=None, lam_poisson=0.3, **kwargs):
        """
        Внести шум на пластину
        :param wafer: np.ndarray: пластина для нанесения паттерна
        :param pattern_mask_array: np.ndarray: маска дефекта
        :param lam_poisson: float: величина лямбды в распределении Пуассона
        :return: noise_wafer, pattern_mask_array: np.ndarray, np.ndarray: пластина с паттерном и маска паттерна
        """
        noise_wafer, pattern_mask = self.noise_poisson(wafer, pattern_mask_array, lam_poisson=lam_poisson)

        return noise_wafer, pattern_mask_array

    def noise_poisson(self, wafer, pattern_mask_array, lam_poisson=0.3):
        """
        Внести случайный шум на карту с помощью пуассоновсокого точечного процесса
        :param wafer: np.ndarray: пластина для нанесения паттерна
        :param pattern_mask_array: np.ndarray: маска дефекта
        :param lam_poisson: float: величина лямбды в распределении Пуассона
        :return: noise_img, pattern_mask_array: np.ndarray, np.ndarray: пластина с паттерном и маска паттерна
        """

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
        for layer in range(pattern_mask_array.shape[0]):
            noise_img[pattern_mask_array[layer, :, :] > self.back_color] = self.defect_color

        return noise_img, pattern_mask_array


def mask_for_visualize(input_mask):
    """
    Форматировать маску с паттернами, записанными по слоям, в маску со всеми паттернами сразу для построения
    :param input_mask: np.ndarray (w, h, dims): маска с паттернами по слоям
    :return: new_mask: np.ndarray (w, h): маска со всеми паттернами на одном слое
    """
    dims, h, w = input_mask.shape
    new_mask = np.zeros((h, w))

    for dim in range(dims):  # пройтись по каждому слою
        color = np.max(input_mask[dim, :, :])  # найти цвет паттерна в слое
        new_mask += input_mask[dim, :, :] / color  # отнормирвать текущий паттерн на свой цвет
    new_mask /= dims  # отнормировать на все паттерны

    return new_mask


def retry(func, *args, **kwargs):
    """
    Функция повторения генератора. Проверка на наличие непустой маски паттерна дефекта.
    Используется для первичного тестирования. Заменена декораторм retry_generator.
    """
    while True:
        wafer, pattern_mask = func(*args, **kwargs)
        if np.count_nonzero(pattern_mask) != 0:
            return wafer, pattern_mask


if __name__ == '__main__':
    """ Проверка функций """
    scratch_generator = ScratchGenerator()  # тестовый генератор
    morph_generator = NoiseGenerator()
    ring_generator = RingGenerator()
    loc_generator = LocGenerator()
    curved_scratch_generator = CurvedScratchGenerator()
    cluster_generator = ClusterGenerator()

    example_count = 20  # примеров для тестирования
    for i in range(example_count):
        fig, maxs = plt.subplots(1, 2, figsize=(10, 7))

        # сгенерировать несколько паттернов на одной пластине
        wafer_map, pattern = None, None
        pattern_count = 1  # количество паттернов на пластине
        for j in range(pattern_count):
            # wafer_map, pattern = scratch_generator(
            #     wafer_map, pattern, line_weight=1, is_noise=True, lam_poisson=1.7)
            # wafer_map, pattern = curved_scratch_generator(wafer_map, pattern, is_noise=True, lam_poisson=1.0)
            wafer_map, pattern = cluster_generator(wafer_map, pattern)
            # wafer_map, pattern = ring_generator(wafer_map, pattern, pattern_type="Donut")
            # wafer_map, pattern = loc_generator(wafer_map, pattern, shape="Rectangle",
            #                            lam_poisson=0.5)
            wafer_map, pattern = morph_generator(wafer_map, pattern)

        # отрисовать результат
        maxs[0].imshow(wafer_map, cmap='inferno')
        maxs[1].imshow(mask_for_visualize(pattern), cmap='inferno')
        plt.show()

    # plt.imshow(create_zero_template_map(IMAGE_DIMS))
    # plt.show()
