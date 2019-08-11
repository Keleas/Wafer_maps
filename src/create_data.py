import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
import cv2
import math
import random

from multiprocessing import cpu_count
from joblib import Parallel, delayed
cpu_count = cpu_count()

IMAGE_DIMS = (92, 92, 1)
wafer_map = pd.read_pickle('test_wafer_map.pkl')
test = cv2.resize(wafer_map.waferMap, dsize=(IMAGE_DIMS[0],IMAGE_DIMS[1]),
                  interpolation=cv2.INTER_NEAREST)

## 2 - паттерн
## 1 - фон
## 0 - область, где нет ничего
test[test == 2] = 1


def sawtooth_line(XC_, YC_, L0_, angle_, line_count, pattern_type, lam_poisson=0.2, save=False):
    size = XC_.shape[0]
    df = [None] * size

    for n in range(size):
        # иниицализация параметров прямой
        L0 = L0_[n]
        XC = XC_[n]
        YC = YC_[n]
        angle = angle_[n]
        ## создадим тестовый полигон
        test = cv2.resize(wafer_map.waferMap, dsize=(IMAGE_DIMS[0], IMAGE_DIMS[1]),
                          interpolation=cv2.INTER_NEAREST)
        ## 2 - паттерн
        ## 1 - фон
        ## 0 - область, где нет ничего
        test[test == 2] = 1
        COLOR_SCALE = 2

        ## параметры уравнения
        def delta_(x, y):
            return int(math.sqrt(x ** 2 + y ** 2))

        delta = np.vectorize(delta_)

        L = L0 - np.sum(delta(XC, YC)[1:])
        N = 200
        x0, y0 = 0, 0

        ## кусочное построение пилообразной прямой
        for i in range(line_count):
            ## случайное удлинение или укорочение отрезка
            import random
            rand = random.randint(-1, 0)
            scale = 0.4
            t = np.linspace(0, L // (line_count + rand * scale), N)

            xc = XC[i]
            yc = YC[i]
            X = np.cos(angle[i]) * t + xc + x0
            Y = np.sin(angle[i]) * t + yc + y0
            X_ = np.around(X)
            Y_ = np.around(Y)

            x_prev, y_prev = x0, y0

            for j in range(X_.shape[0]):
                x = int(X_[j])
                y = int(Y_[j])
                if j == 0:
                    ## первая точка прямой
                    x_first, y_first = x, y

                try:
                    if test[x, y] == 1:
                        test[x, y] = COLOR_SCALE
                        x0, y0 = x, y
                except IndexError:
                    break

            ## сшивка прямых
            if i != 0:
                ## уравнение прямой сшивки
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
                        if test[x, y] == 1:
                            test[x, y] = COLOR_SCALE
                    except IndexError:
                        break

        #         kernel = np.ones((3,3), np.uint8)
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        test = cv2.dilate(test, kernel, iterations=1)
        test = cv2.morphologyEx(test, cv2.MORPH_CLOSE, kernel)

        noise_img = test.copy()
        mask = np.random.randint(0, 2, size=noise_img.shape).astype(np.bool)
        mask[noise_img == 0] = False
        r = np.random.poisson(lam=lam_poisson, size=noise_img.shape)
        r[r == 0] = 1
        r[r > 2] = 2
        noise_img[mask] = r[mask]

        #         kernel = np.ones((3,3), np.uint8)
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        noise_img = cv2.morphologyEx(noise_img, cv2.MORPH_CLOSE, kernel)

        df[n] = [noise_img, pattern_type]

        ## для презенташки
        if save == True:
            path = 'output/test_classes/{}'.format(pattern_type)
            try:
                os.mkdir(path)
            except OSError:
                pass
            plt.imshow(noise_img, cmap='inferno')
            name = '/{}{}.jpg'.format(pattern_type, n)
            plt.savefig(path + name)

    return pd.DataFrame(df, columns=['waferMap', 'failureType'])


def create_rings(XC, YC, R_, PHI, N, pattern_type, lam_poisson=1.2, save=False):
    size = XC.shape[0]
    df = [None] * size

    for n in range(size):
        ## параметры кольца
        phi = np.linspace(PHI[n][0], PHI[n][1], N[n])
        r = np.linspace(R_[n][0], R_[n][1], N[n])
        xc = XC[n]
        yc = YC[n]

        ## синтез сетки
        R, Fi = np.meshgrid(r, phi)
        X = R * (np.cos(Fi)) + xc
        Y = R * (np.sin(Fi)) + yc
        X_ = np.around(X)
        Y_ = np.around(Y)

        ## индексы для полигона
        points = []
        for i in range(X_.shape[0]):
            for j in range(X_.shape[1]):
                x = X_[i, j]
                y = Y_[i, j]
                points.append((x, y))

        ## создадим тестовый полигон
        test = cv2.resize(wafer_map.waferMap, dsize=(IMAGE_DIMS[0], IMAGE_DIMS[1]),
                          interpolation=cv2.INTER_NEAREST)
        ## 2 - паттерн
        ## 1 - фон
        ## 0 - область, где нет ничего
        test[test == 2] = 1
        COLOR_SCALE = 2

        for idx in points:
            i, j = idx
            i = int(round(i))
            j = int(round(j))
            try:
                if test[i, j] == 1:
                    test[i, j] = COLOR_SCALE
            except IndexError:
                break

        is_dilate = random.randint(-1, 1)
        if is_dilate == 1:
            ## сверткой расширим
            kernel = np.ones((3, 3), np.uint8)
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            count_iter = random.randint(1, 3)
            test = cv2.dilate(test, kernel, iterations=count_iter)
            test = cv2.morphologyEx(test, cv2.MORPH_CLOSE, kernel)

        ## внесем шум
        noise_img = test.copy()
        mask = np.random.randint(0, 2, size=noise_img.shape).astype(np.bool)
        mask[noise_img == 0] = False
        r = np.random.poisson(lam=lam_poisson, size=noise_img.shape)
        # нормировка на шумы
        r[r == 0] = 1
        r[r > 2] = 2
        noise_img[mask] = r[mask]

        ## сверткой расширим
        kernel = np.ones((3, 3), np.uint8)
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        noise_img = cv2.morphologyEx(noise_img, cv2.MORPH_CLOSE, kernel)
        noise_img = cv2.erode(noise_img, kernel, iterations=1)

        df[n] = [noise_img, pattern_type]

        ## для презенташки
        if save == True:
            path = 'output/test_classes/{}'.format(pattern_type)
            try:
                os.mkdir(path)
            except OSError:
                pass
            plt.imshow(noise_img, cmap='inferno')
            name = '/{}{}.jpg'.format(pattern_type, n)
            plt.savefig(path + name)

    return pd.DataFrame(df, columns=['waferMap', 'failureType'])


def read_data(path):
    df_all_standart = pd.read_pickle(os.path.join(os.getcwd(), path))

    return df_all_standart

read_data(path='input\LSWMD.pkl')