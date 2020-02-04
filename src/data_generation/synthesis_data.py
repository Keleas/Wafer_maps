import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool

print(os.getcwd())

from src.data_generation.pattern_generators import (
    ScratchGenerator,
    RingGenerator,
    NoiseGenerator,
    mask_for_visualize
)


def synthesis_scratch(wafer_count):
    """
    Синтезирование базы данных из паттерна "Scratch" для обучающей выборки.
    ВАЖНО - для количество конфигураций и их тип определяется вручную.
    :param wafer_count: int: количество синтезированных вафлей при одной конфигурации параметров модели
    :return: pandas.DataFrame:
    """
    # общее количество синтезированных данных
    # определяется wafer_count и количеством конфигураций
    total_wafer_count = wafer_count * 3

    generator = ScratchGenerator()  # генератор паттерна
    noise = NoiseGenerator()  # генератор шума
    synthesis_base = []  # общая выборка
    """ Конфигурация """
    for _ in tqdm(range(wafer_count), ascii=True, total=total_wafer_count,
                  desc='line_weight=1, lam=1.7'):
        wafer, mask = generator(line_weight=1, is_noise=True, lam_poisson=1.7)
        wafer, mask = noise(wafer, mask)
        synthesis_base.append([wafer, mask])

    """ Конфигурация """
    for _ in tqdm(range(wafer_count), ascii=True, total=total_wafer_count/2,
                  desc='line_weight=2, lam=2.1'):
        wafer, mask = generator(line_weight=2, is_noise=True, lam_poisson=2.1)
        wafer, mask = noise(wafer, mask)
        synthesis_base.append([wafer, mask])

    """ Конфигурация """
    for _ in tqdm(range(wafer_count), ascii=True, total=total_wafer_count/3,
                  desc='line_weight=3, lam=0.7'):
        wafer, mask = generator(line_weight=3, is_noise=True, lam_poisson=0.7)
        wafer, mask = noise(wafer, mask)
        synthesis_base.append([wafer, mask])

    # преобразовать list в pandas.DataFrame
    database = pd.DataFrame(synthesis_base, columns=['wafer', 'mask'])
    # сохарнить базу данных в формат csv
    database.to_csv('../../input/synthesis/test_database.csv')

    return database


if __name__ == '__main__':
    """ Визуальная проверка сгенерированных данных """
    scratch_database = synthesis_scratch(wafer_count=100)

    number_example = 50  # количество случайных примеров
    for i in range(number_example):
        sample = scratch_database.sample(n=1)  # взять случайный элемент из базы
        # выделить вафлю и маску из sample
        wafer, mask = sample['wafer'].values[0], sample['mask'].values[0]

        # отрисовать примеры
        fig, maxs = plt.subplots(1, 2, figsize=(8, 8))
        maxs = maxs.ravel(order='C')
        maxs[0].imshow(wafer, cmap='inferno')
        maxs[1].imshow(mask_for_visualize(mask), cmap='inferno')
        plt.show()
