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

    # The two lines below could be merged, but I stored the mask
    # for code clarity.
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


class ScratchGenerator(BasisGenerator):
    """ Класс для добавления паттерна "Scratch" на пластину """

    def __init__(self):
        super(ScratchGenerator, self).__init__()
        self.pattern_color = 10  # цвет паттерна Scratch

    def __call__(self, wafer=None, mask=None,
                 length=None, line_weight=None, all_xc=None, all_yc=None, all_angle=None, part_line_count=None):
        """
        Сгенерировать паттерн "Scratch" на заданной пластине.
        :param wafer: np.ndarray: пластина для нанесения паттерна
        :param mask: np.ndarray: маска дефекта
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
                        #_slice = slice(_x - low_weight: _x + high_weight,  _y - low_weight:_y + high_weight)
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

        # временное решение — замена морфологических операций
        # формируем пуассоновкие точки от 0 до 1
        random_poisson = np.random.poisson(lam=0.7, size=pattern_mask.shape)
        random_poisson = random_poisson / np.amax(random_poisson)

        # накладываем на объект и очищаем малые значения
        pattern_mask = pattern_mask + random_poisson
        pattern_mask[pattern_mask <= self.pattern_color] = self.back_color
        pattern_mask[pattern_mask > self.pattern_color] = self.pattern_color - self.wafer_color

        # вырезаем оригинальный паттерн и накладываемый новый "шумный"
        wafer[wafer == self.pattern_color] = self.wafer_color
        wafer = wafer + pattern_mask

        if mask is None:
            mask = np.expand_dims(pattern_mask, axis=2)
        else:
            mask = np.dstack((mask, pattern_mask))

        wafer[wafer == self.pattern_color] = self.defect_color  # нормировка значений

        return wafer, mask


class PoisonGenerator(BasisGenerator):
    """ Класс для добавления Пуассоновского точечного процесса на пластину """

    def __init__(self):
        super(PoisonGenerator, self).__init__()
        self.pattern_color = 4

    # TODO: заложить два функционала:
    #       1) случайный локальный процесс вдоль заданной маски для каждого слоя паттерна
    #       2) случайны локальный процесс на всей пластине
    def __call__(self, wafer=None, mask=None,
                 *args, **kwargs):
        h, w, dims = mask.shape

        # по каждому слою паттерна
        for dim in range(dims):
            mask_layer = mask[:, :, dim]
            points = np.argwhere(mask_layer > 0)

            window_step = 3
            window_size = 5
            lam_poisson = 2.1
            layer_color = np.max(mask_layer)
            for x, y in points[::5]:
                try:
                    window = mask[abs(y-window_size):y+window_size, abs(x-window_size):x+window_size]

                    noise_img = window.copy()
                    noise_mask = np.random.randint(0, 2, size=noise_img.shape).astype(np.bool)
                    noise_mask[noise_img == 0] = False
                    r = np.random.poisson(lam=lam_poisson, size=noise_img.shape)
                    # нормировка на величину шума
                    # print(r)
                    # r[r == self.back_color] = self.wafer_color
                    r[r > self.wafer_color] = layer_color
                    noise_img[noise_mask] = r[noise_mask]

                    mask[abs(y - window_size):y + window_size, abs(x - window_size):x + window_size] = noise_img
                except IndexError:
                    continue

        return wafer, mask


class MorphNoiseGenerator(BasisGenerator):
    """ Класс для добавления шумов на пластину """

    def __init__(self):
        super(MorphNoiseGenerator, self).__init__()
        self.pattern_color = 5

    # TODO: перенести функцию добавления шума с помощью морфологических операций для каждого слоя паттерна
    def __call__(self, wafer=None, mask=None,
                 *args, **kwargs):
        noise_wafer, noise_mask = self.noise_simple(wafer, mask, dilate_time=1)

        return noise_wafer, noise_mask

    @staticmethod
    def noise_simple(wafer, mask, dilate_time=1):
        kernel2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        count_iter = random.randint(1, dilate_time)
        wafer = cv2.dilate(wafer, kernel2, iterations=count_iter)
        mask = cv2.dilate(mask, kernel2, iterations=count_iter)

        return wafer, mask


def mask_for_visualize(mask):
    """
    Форматировать маску с паттернами, записанными по слоям, в маску со всеми паттернами сразу для построения
    :param mask: np.ndarray (w, h, dims): маска с паттернами по слоям
    :return: np.ndarray (w, h): маска со всеми паттернами на одном слое
    """
    h, w, dims = mask.shape
    new_mask = np.zeros((h, w))

    for dim in range(dims):  # пройтись по каждому слою
        color = np.max(mask[:, :, dim])  # найти цвет паттерна в слое
        new_mask += mask[:, :, dim] / color  # отнормирвать текущий паттерн на свой цвет
    new_mask /= dims  # отнормировать на все паттерны

    return new_mask


if __name__ == '__main__':
    """ Проверка функций """
    scratch_generator = ScratchGenerator()  # тестовый генератор
    morph_generator = MorphNoiseGenerator()
    poison_generator = PoisonGenerator()

    example_count = 1  # примеров для тестирования
    for i in range(example_count):
        fig, maxs = plt.subplots(1, 2, figsize=(10, 7))

        # сгенерировать несколько паттернов на одной пластине
        wafer, mask = None, None
        pattern_count = 4  # количество паттернов на пластине
        for i in range(pattern_count):
            wafer, mask = scratch_generator(wafer, mask)
            wafer, mask = poison_generator(wafer, mask)
            # wafer, mask = morph_generator(wafer, mask)

        # отрисовать результат
        maxs[0].imshow(wafer, cmap='inferno')
        maxs[1].imshow(mask_for_visualize(mask), cmap='inferno')
        plt.show()

    # plt.imshow(create_zero_template_map(IMAGE_DIMS))
    # plt.show()
