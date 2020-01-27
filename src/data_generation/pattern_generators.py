import os
import time
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")


class BasisGenerator(object):
    """Класс шаблон для генератора паттерна"""

    # TODO: инициализация пустого паттерна и параметров пластины
    def __init__(self):
        pass

    # TODO: вернуть пустой паттерн
    def __call__(self, *args, **kwargs):
        pass


class ScratchGenerator(BasisGenerator):
    """Класс для добавления паттерна Царапина на пластину"""
    def __init__(self):
        super(ScratchGenerator, self).__init__()

    # TODO: функционал класса:
    #       1) генерация паттерна на заданном шаблоне с заданными параметрами модели
    #       2) генерация паттерна на чистом шаблоне с незаданными (сгенерировать самому) параметрами модели
    def __call__(self, img, *args, **kwargs):
        pass


class PoisonGenerator(BasisGenerator):
    """Класс для добавления Пуассоновского точечного процесса на пластину"""
    def __init__(self):
        super(PoisonGenerator, self).__init__()

    # TODO: заложить два функционала:
    #       1) случайный локальный процесс вдоль заданной маски паттерна
    #       2) случайны локальный процесс на всей пластине
    def __call__(self, *args, **kwargs):
        pass


class MorphNoiseGenerator(BasisGenerator):
    """Класс для добавления шумов на пластину"""
    def __init__(self):
        super(MorphNoiseGenerator, self).__init__()

    # TODO: перенести функцию добавления шума с помощью морфологических операций
    def __call__(self, *args, **kwargs):
        pass


if __name__ == '__main__':
    pass
