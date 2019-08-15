import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
import math
import random
from tqdm import tqdm
import gc
import time
from sklearn.model_selection import train_test_split
from multiprocessing import cpu_count
from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")


class SynthesizedDatabaseCreator(object):
    """

    """
    def __init__(self):
        self.IMAGE_DIMS = (96, 96, 1)
        self.number_points = 100
        self.cpu_count = cpu_count() - 1

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
        self.template_map = load_template_map(self.IMAGE_DIMS)

    def sawtooth_line(self, XC_, YC_, L0_, angle_, line_count, pattern_type, lam_poisson=0.2, save=False):
        """

        :param XC_:
        :param YC_:
        :param L0_:
        :param angle_:
        :param line_count:
        :param pattern_type:
        :param lam_poisson:
        :param save:
        :return:
        """
        size = XC_.shape[0]
        synthesized_base = [None] * size

        for n in tqdm(range(size)):
            # иниицализация параметров прямой
            L0 = L0_[n]
            XC = XC_[n]
            YC = YC_[n]
            angle = angle_[n]

            # параметры уравнения
            def delta_(x, y):
                return int(math.sqrt(x ** 2 + y ** 2))
            delta = np.vectorize(delta_)
            L = L0 - np.sum(delta(XC, YC)[1:])
            N = 200
            x0, y0 = 0, 0

            # тестовый полигон
            template = self.template_map.copy()
            COLOR_SCALE = 2

            # кусочное построение пилообразной прямой
            for i in range(line_count):
                # случайное удлинение или укорочение отрезка
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

                x_first, y_first = 0, 0
                for j in range(X_.shape[0]):
                    x = int(X_[j])
                    y = int(Y_[j])
                    if j == 0:
                        # первая точка прямой
                        x_first, y_first = x, y
                    try:
                        if template[x, y] == 1:
                            template[x, y] = COLOR_SCALE
                            x0, y0 = x, y
                    except IndexError:
                        break

                # сшивка прямых
                if i != 0:
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
                            if template[x, y] == 1:
                                template[x, y] = COLOR_SCALE
                        except IndexError:
                            break

            # kernel = np.ones((3,3), np.uint8)
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            template = cv2.dilate(template, kernel, iterations=1)
            template = cv2.morphologyEx(template, cv2.MORPH_CLOSE, kernel)

            noise_img = template.copy()
            mask = np.random.randint(0, 2, size=noise_img.shape).astype(np.bool)
            mask[noise_img == 0] = False
            r = np.random.poisson(lam=lam_poisson, size=noise_img.shape)
            r[r == 0] = 1
            r[r > 2] = 2
            noise_img[mask] = r[mask]

            #  kernel = np.ones((3,3), np.uint8)
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            noise_img = cv2.morphologyEx(noise_img, cv2.MORPH_CLOSE, kernel)

            synthesized_base[n] = [noise_img, pattern_type]
            # для презентации
            if save:
                path = 'output/test_classes/{}'.format(pattern_type)
                try:
                    os.mkdir(path)
                except OSError:
                    pass
                plt.imshow(noise_img, cmap='inferno')
                name = '/{}{}.jpg'.format(pattern_type, n)
                plt.savefig(path + name)

        return pd.DataFrame(synthesized_base, columns=['waferMap', 'failureType'])

    def generator_scratch(self, mode=0, plot=False):
        """

        :param mode:
        :param plot:
        :return:
        """
        print('[INFO] Create scratches')
        # число синтезированных карт
        N_POINTS = self.number_points

        # суммарная длина отрезка
        L0 = np.random.randint(0.4 * self.IMAGE_DIMS[0], 0.65 * self.IMAGE_DIMS[0], size=N_POINTS)

        # X координата старта прямой
        XC = np.random.randint(0.2 * self.IMAGE_DIMS[0], 0.4 * self.IMAGE_DIMS[0], size=N_POINTS)
        # смещение по x для старта следующей прямой
        delta_xc = np.random.randint(0.02 * self.IMAGE_DIMS[0], 0.05 * self.IMAGE_DIMS[0], size=N_POINTS)
        np.random.shuffle(delta_xc)
        XC = np.vstack((XC, delta_xc))
        # merge под формат генератора
        xc = np.array([[XC[0, i], XC[1, i]] for i in range(XC.shape[1])])

        # Y координата старта прямой
        YC = np.random.randint(0.5 * self.IMAGE_DIMS[0], 0.7 * self.IMAGE_DIMS[0], size=N_POINTS)
        # смещение по x для старта следующей прямой
        delta_yc = np.random.randint(0.04 * self.IMAGE_DIMS[0], 0.09 * self.IMAGE_DIMS[0], size=N_POINTS)
        np.random.shuffle(delta_yc)
        YC = np.vstack((YC, delta_yc))
        # merge под формат генератора
        yc = np.array([[YC[0, i], YC[1, i]] for i in range(YC.shape[1])])

        # углы наклона для каждого отрезка
        angle1 = np.random.randint(-34, -10, size=N_POINTS) * np.pi / 180
        angle2 = np.random.randint(-34, -25, size=N_POINTS) * np.pi / 180
        angle = np.vstack((angle1, angle2))
        angle = np.array([[angle[0, i], angle[1, i]] for i in range(angle.shape[1])])

        PHI = None
        for i in range(4):
            # угол старта для сектора
            phi1 = np.random.uniform(45 * i, 30 * (i + 1), size=N_POINTS // 4) * np.pi / 180
            # угол конца для сектора
            phi2 = np.random.uniform(30 * (i + 1), 60 * (i + 1), size=N_POINTS // 4) * np.pi / 180
            phi = np.vstack((phi1, phi2))
            # merge под формат генератора
            phi = np.array([[phi[0, j], phi[1, j]] for j in range(phi.shape[1])])
            if i == 0:
                PHI = phi
            else:
                PHI = np.vstack((PHI, phi))

        df_scratch_curved = None
        if mode == 1:
            # генератор для презенташки
            df_scratch_curved = self.sawtooth_line(xc, yc, L0, angle,
                                                   pattern_type='Scratch', line_count=2,
                                                   lam_poisson=0.7, save=False)
        elif mode == 0:
            # генератор параллельный
            n_workers = self.cpu_count
            results = Parallel(n_workers)(
                delayed(self.sawtooth_line)(xc[i::n_workers], yc[i::n_workers],
                                            L0[i::n_workers], angle[i::n_workers],
                                            pattern_type='Scratch', line_count=2)
                for i in range(n_workers))

            df_scratch_curved = results[0]
            for i in range(1, len(results)):
                df_scratch_curved = pd.concat((df_scratch_curved, results[i]), sort=False)

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, id in enumerate(sample_idx):
                ax[i].imshow(df_scratch_curved.waferMap.values[id], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_scratch_curved

    def create_rings(self, XC, YC, R_, PHI, N, pattern_type, lam_poisson=1.2, save=False):
        """

        :param YC:
        :param R_:
        :param PHI:
        :param N:
        :param pattern_type:
        :param lam_poisson:
        :param save:
        :return:
        """
        size = XC.shape[0]
        synthesized_base = [None] * size

        for n in tqdm(range(size)):
            # параметры кольца
            phi = np.linspace(PHI[n][0], PHI[n][1], N[n])
            r = np.linspace(R_[n][0], R_[n][1], N[n])
            xc = XC[n]
            yc = YC[n]

            # синтез сетки
            R, Fi = np.meshgrid(r, phi)
            X = R * (np.cos(Fi)) + xc
            Y = R * (np.sin(Fi)) + yc
            X_ = np.around(X)
            Y_ = np.around(Y)

            # индексы для полигона
            points = []
            for i in range(X_.shape[0]):
                for j in range(X_.shape[1]):
                    x = X_[i, j]
                    y = Y_[i, j]
                    points.append((x, y))

            # тестовый полигон
            template = self.template_map.copy()
            COLOR_SCALE = 2

            for idx in points:
                i, j = idx
                i = int(round(i))
                j = int(round(j))
                try:
                    if template[i, j] == 1:
                        template[i, j] = COLOR_SCALE
                except IndexError:
                    break

            is_dilate = random.randint(-1, 1)
            if is_dilate == 1:
                # сверткой расширим
                kernel = np.ones((3, 3), np.uint8)
                kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
                count_iter = random.randint(1, 3)
                template = cv2.dilate(template, kernel, iterations=count_iter)
                template = cv2.morphologyEx(template, cv2.MORPH_CLOSE, kernel)

            # внесем шум
            noise_img = template.copy()
            mask = np.random.randint(0, 2, size=noise_img.shape).astype(np.bool)
            mask[noise_img == 0] = False
            r = np.random.poisson(lam=lam_poisson, size=noise_img.shape)
            # нормировка на шумы
            r[r == 0] = 1
            r[r > 2] = 2
            noise_img[mask] = r[mask]

            # сверткой расширим
            # kernel = np.ones((3, 3), np.uint8)
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            noise_img = cv2.morphologyEx(noise_img, cv2.MORPH_CLOSE, kernel)
            noise_img = cv2.erode(noise_img, kernel, iterations=1)

            synthesized_base[n] = [noise_img, pattern_type]

            # для презентации
            if save:
                path = 'output/test_classes/{}'.format(pattern_type)
                try:
                    os.mkdir(path)
                except OSError:
                    pass
                plt.imshow(noise_img, cmap='inferno')
                name = '/{}{}.jpg'.format(pattern_type, n)
                plt.savefig(path + name)

        return pd.DataFrame(synthesized_base, columns=['waferMap', 'failureType'])

    def generator_rings(self, mode=0, plot=False):
        """

        :param mode:
        :param plot:
        :return:
        """
        print('[INFO] Create rings')
        # число синтезированных карт
        N_POINTS = self.number_points

        PHI = None
        for i in range(4):
            # угол старта для сектора
            phi1 = np.random.uniform(0 + 95 * i, 30 + 95 * i, size=N_POINTS // 4) * np.pi / 180
            # угол конца для сектора
            phi2 = np.random.uniform(320 + 90 * i, 360 * (i + 1), size=N_POINTS // 4) * np.pi / 180
            phi = np.vstack((phi1, phi2))
            # merge под формат генератора
            phi = np.array([[phi[0, j], phi[1, j]] for j in range(phi.shape[1])])
            if i == 0:
                PHI = phi
            else:
                PHI = np.vstack((PHI, phi))

        # радиус внутреннего круга
        r1 = np.random.randint(0.24 * self.IMAGE_DIMS[0], 0.26 * self.IMAGE_DIMS[0], size=N_POINTS)
        # радиус внешнего круга
        r2 = np.random.randint(0.3 * self.IMAGE_DIMS[0], 0.35 * self.IMAGE_DIMS[0], size=N_POINTS)
        r = np.vstack((r1, r2))
        # merge под формат генератора
        r = np.array([[r[0, i], r[1, i]] for i in range(r.shape[1])])

        # X координата старта прямой
        XC = np.random.randint(0.22 * self.IMAGE_DIMS[0], 0.65 * self.IMAGE_DIMS[0], size=N_POINTS)
        # Y координата старта прямой
        YC = np.random.randint(0.22 * self.IMAGE_DIMS[0], 0.65 * self.IMAGE_DIMS[0], size=N_POINTS)

        # интесивность
        N = np.random.randint(200, 210, size=N_POINTS)

        df_donut = None
        if mode == 1:
            # генератор для презенташки
            df_donut = self.create_rings(XC, YC, r, PHI, N, pattern_type='Donut', save=True, lam_poisson=1.7)

        elif mode == 0:
            # генератор параллельный
            n_workers = self.cpu_count
            results = Parallel(n_workers)(
                delayed(self.create_rings)(XC[i::n_workers], YC[i::n_workers],
                                      r[i::n_workers], PHI[i::n_workers],
                                      N[i::n_workers], pattern_type='Donut')
                for i in range(n_workers))

            df_donut = results[0]
            for i in range(1, len(results)):
                df_donut = pd.concat((df_donut, results[i]), sort=False)

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, id in enumerate(sample_idx):
                ax[i].imshow(df_donut.waferMap.values[id], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_donut

    def generator_loc(self, mode=0, plot=False):
        """

        :param mode:
        :param plot:
        :return:
        """
        print('[INFO] Create loc')
        # число синтезированных карт
        N_POINTS = self.number_points

        PHI = None
        for i in range(4):
            # угол старта для сектора
            phi1 = np.random.uniform(95 * i, 55 + 90 * i, size=N_POINTS // 4) * np.pi / 180
            # угол конца для сектора
            phi2 = np.random.uniform(65 + 90 * i, 95 * (i + 1), size=N_POINTS // 4) * np.pi / 180
            phi = np.vstack((phi1, phi2))
            # merge под формат генератора
            phi = np.array([[phi[0, j], phi[1, j]] for j in range(phi.shape[1])])
            if i == 0:
                PHI = phi
            else:
                PHI = np.vstack((PHI, phi))

        # радиус внутреннего круга
        r1 = np.random.randint(0.1 * self.IMAGE_DIMS[0], 0.2 * self.IMAGE_DIMS[0], size=N_POINTS)
        # радиус внешнего круга
        r2 = np.random.randint(0.2 * self.IMAGE_DIMS[0], 0.25 * self.IMAGE_DIMS[0], size=N_POINTS)
        r = np.vstack((r1, r2))
        # merge под формат генератора
        r = np.array([[r[0, i], r[1, i]] for i in range(r.shape[1])])

        # X координата старта прямой
        XC = np.random.randint(0.45 * self.IMAGE_DIMS[0], 0.55 * self.IMAGE_DIMS[0], size=N_POINTS)
        # Y координата старта прямой
        YC = np.random.randint(0.45 * self.IMAGE_DIMS[0], 0.55 * self.IMAGE_DIMS[0], size=N_POINTS)

        # интесивность
        N = np.random.randint(200, 210, size=N_POINTS)

        df_loc = None
        if mode == 1:
            # генератор для презенташки
            df_loc = self.create_rings(XC, YC, r, PHI, N, pattern_type='Loc', save=True)
        elif mode == 0:
            # генератор параллельный
            n_workers = self.cpu_count
            results = Parallel(n_workers)(
                delayed(self.create_rings)(XC[i::n_workers], YC[i::n_workers],
                                      r[i::n_workers], PHI[i::n_workers],
                                      N[i::n_workers], pattern_type='Loc')
                for i in range(n_workers))

            df_loc = results[0]
            for i in range(1, len(results)):
                df_loc = pd.concat((df_loc, results[i]), sort=False)

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, id in enumerate(sample_idx):
                ax[i].imshow(df_loc.waferMap.values[id], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_loc

    def generator_center(self, mode=0, plot=False):
        """

        :param mode:
        :param plot:
        :return:
        """
        print('[INFO] Create center')
        # число синтезированных карт
        N_POINTS = self.number_points

        PHI = None
        for i in range(4):
            # угол старта для сектора
            phi1 = np.random.uniform(95 * i, 10 + 90 * i, size=N_POINTS // 4) * np.pi / 180
            # угол конца для сектора
            phi2 = np.random.uniform(45 + 90 * i, 95 * (i + 1), size=N_POINTS // 4) * np.pi / 180
            phi = np.vstack((phi1, phi2))
            # merge под формат генератора
            phi = np.array([[phi[0, j], phi[1, j]] for j in range(phi.shape[1])])
            if i == 0:
                PHI = phi
            else:
                PHI = np.vstack((PHI, phi))

        # радиус внутреннего круга
        r1 = np.random.randint(0.0 * self.IMAGE_DIMS[0], 0.05 * self.IMAGE_DIMS[0], size=N_POINTS)
        # радиус внешнего круга
        r2 = np.random.randint(0.12 * self.IMAGE_DIMS[0], 0.23 * self.IMAGE_DIMS[0], size=N_POINTS)
        r = np.vstack((r1, r2))
        # merge под формат генератора
        r = np.array([[r[0, i], r[1, i]] for i in range(r.shape[1])])

        # X координата старта прямой
        XC = np.random.randint(0.48 * self.IMAGE_DIMS[0], 0.5 * self.IMAGE_DIMS[0], size=N_POINTS)
        # Y координата старта прямой
        YC = np.random.randint(0.48 * self.IMAGE_DIMS[0], 0.5 * self.IMAGE_DIMS[0], size=N_POINTS)

        # интесивность
        N = np.random.randint(200, 210, size=N_POINTS)

        if mode == 1:
            # генератор для презенташки
            df_center = self.create_rings(XC, YC, r, PHI, N, pattern_type='Center', save=True)
        elif mode == 0:
            # генератор параллельный
            n_workers = self.cpu_count
            results = Parallel(n_workers)(
                delayed(self.create_rings)(XC[i::n_workers], YC[i::n_workers],
                                      r[i::n_workers], PHI[i::n_workers],
                                      N[i::n_workers], pattern_type='Center')
                for i in range(n_workers))

            df_center = results[0]
            for i in range(1, len(results)):
                df_center = pd.concat((df_center, results[i]), sort=False)

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, id in enumerate(sample_idx):
                ax[i].imshow(df_center.waferMap.values[id], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_center

    def generator_edge_ring(self, mode=0, plot=False):
        """

        :param mode:
        :param plot:
        :return:
        """
        print('[INFO] Create edge_ring')
        # число синтезированных карт
        N_POINTS = self.number_points

        PHI = None
        for i in range(4):
            # угол старта для сектора
            phi1 = np.random.uniform(0 + 90 * i, 30 + 90 * i, size=N_POINTS // 4) * np.pi / 180
            # угол конца для сектора
            phi2 = np.random.uniform(320 + 90 * i,
                                     360 * (i + 1), size=N_POINTS // 4) * np.pi / 180
            phi = np.vstack((phi1, phi2))
            # merge под формат генератора
            phi = np.array([[phi[0, j], phi[1, j]] for j in range(phi.shape[1])])
            if i == 0:
                PHI = phi
            else:
                PHI = np.vstack((PHI, phi))

        center = 0.5 * self.IMAGE_DIMS[0]
        r1 = np.random.randint(center - 4, center - 3, size=N_POINTS)
        r2 = np.random.randint(center, center + 1, size=N_POINTS)
        r = np.vstack((r1, r2))
        r = np.array([[r[0, i], r[1, i]] for i in range(r.shape[1])])

        # X координата старта прямой
        XC = np.random.randint(center - 2, center, size=N_POINTS)
        # Y координата старта прямой
        YC = np.random.randint(center - 2, center, size=N_POINTS)

        # интесивность
        N = np.random.randint(200, 210, size=N_POINTS)

        df_edge_ring = None
        if mode == 1:
            # генератор для презенташки
            df_edge_ring = self.create_rings(XC, YC, r, PHI, N, pattern_type='Edge-Ring', save=True)
        elif mode == 0:
            # генератор параллельный
            n_workers = self.cpu_count
            results = Parallel(n_workers)(
                delayed(self.create_rings)(XC[i::n_workers], YC[i::n_workers],
                                      r[i::n_workers], PHI[i::n_workers],
                                      N[i::n_workers], pattern_type='Edge-Ring')
                for i in range(n_workers))

            df_edge_ring = results[0]
            for i in range(1, len(results)):
                df_edge_ring = pd.concat((df_edge_ring, results[i]), sort=False)

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, id in enumerate(sample_idx):
                ax[i].imshow(df_edge_ring.waferMap.values[id], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_edge_ring

    def generator_edge_loc(self, mode=0, plot=False):
        """

        :param mode:
        :param plot:
        :return:
        """
        print('[INFO] Create edge_ring')
        # число синтезированных карт
        N_POINTS = self.number_points

        PHI = None
        for i in range(4):
            # угол старта для сектора
            phi1 = np.random.uniform(15 + 90 * i, 25 + 90 * i, size=N_POINTS // 4) * np.pi / 180
            # угол конца для сектора
            phi2 = np.random.uniform(55 + 90 * i, 115 + 90 * i, size=N_POINTS // 4) * np.pi / 180
            phi = np.vstack((phi1, phi2))
            # merge под формат генератора
            phi = np.array([[phi[0, j], phi[1, j]] for j in range(phi.shape[1])])
            if i == 0:
                PHI = phi
            else:
                PHI = np.vstack((PHI, phi))

        center = 0.5 * self.IMAGE_DIMS[0]
        r1 = np.random.randint(center - 5, center - 3, size=N_POINTS)
        r2 = np.random.randint(center, center + 1, size=N_POINTS)
        r = np.vstack((r1, r2))
        r = np.array([[r[0, i], r[1, i]] for i in range(r.shape[1])])

        # X координата старта прямой
        XC = np.random.randint(center - 2, center - 1, size=N_POINTS)
        # Y координата старта прямой
        YC = np.random.randint(center - 2, center - 1, size=N_POINTS)

        # интесивность
        N = np.random.randint(200, 210, size=N_POINTS)

        df_edge_loc = None
        if mode == 1:
            # генератор для презенташки
            df_edge_loc = self.create_rings(XC, YC, r, PHI, N, pattern_type='Edge-Loc', save=True)
        elif mode == 0:
            # генератор параллельный
            n_workers = self.cpu_count
            results = Parallel(n_workers)(
                delayed(self.create_rings)(XC[i::n_workers], YC[i::n_workers],
                                      r[i::n_workers], PHI[i::n_workers],
                                      N[i::n_workers], pattern_type='Edge-Loc')
                for i in range(n_workers))

            df_edge_loc = results[0]
            for i in range(1, len(results)):
                df_edge_loc = pd.concat((df_edge_loc, results[i]), sort=False)

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, id in enumerate(sample_idx):
                ax[i].imshow(df_edge_loc.waferMap.values[id], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_edge_loc

    def create_near_full(self, N, pattern_type, lam_poisson=1.2, save=False):
        """

        :param pattern_type:
        :param lam_poisson:
        :param save:
        :return:
        """
        synthesized_base = [None] * N
        for n in range(N):
            # тестовый полигон
            template = self.template_map
            COLOR_SCALE = 5

            # внесем шум
            noise_img = template.copy()
            mask = np.random.randint(0, 2, size=noise_img.shape).astype(np.bool)
            mask[noise_img == 0] = False
            r = np.random.poisson(lam=lam_poisson, size=noise_img.shape)
            # нормировка на шумы
            # r = np.around(r//np.max(r))
            r[r == 0] = 1
            r[r == 1] = 2
            r[r > 2] = 1
            noise_img[mask] = r[mask]

            ## сверткой расширим
            kernel = np.ones((3, 3), np.uint8)
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            noise_img = cv2.morphologyEx(noise_img, cv2.MORPH_CLOSE, kernel)
            noise_img = cv2.erode(noise_img, kernel, iterations=1)

            synthesized_base[n] = [noise_img, pattern_type]

            ## для презенташки
            if save:
                path = 'output/test_classes/{}'.format(pattern_type)
                try:
                    os.mkdir(path)
                except OSError:
                    pass
                plt.imshow(noise_img, cmap='inferno')
                name = '/{}{}.jpg'.format(pattern_type, n)
                plt.savefig(path + name)

        return pd.DataFrame(synthesized_base, columns=['waferMap', 'failureType'])

    def generator_near_full(self, plot=False):
        """

        :param plot:
        :return:
        """
        print('[INFO] Create near_full')
        # число синтезированных карт
        N_POINTS = self.number_points

        df_near_full = self.create_near_full(N_POINTS, pattern_type='Near-full', lam_poisson=1.3)

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, id in enumerate(sample_idx):
                ax[i].imshow(df_near_full.waferMap.values[id], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_near_full

    def create_random(self, N, pattern_type, lam_poisson=1.2, save=False):
        """

        :param N:
        :param pattern_type:
        :param lam_poisson:
        :param save:
        :return:
        """
        synthesized_base = [None] * N
        for n in tqdm(range(N)):
            # тестовый полигон
            template = self.template_map
            COLOR_SCALE = 5

            ## внесем шум
            noise_img = template.copy()
            mask = np.random.randint(0, 2, size=noise_img.shape).astype(np.bool)
            mask[noise_img == 0] = False
            r = np.random.poisson(lam=lam_poisson, size=noise_img.shape)
            # нормировка на шумы
            # r = np.around(r//np.max(r))
            r[r == 0] = 1
            r[r > 2] = 2
            noise_img[mask] = r[mask]

            ## сверткой расширим
            kernel = np.ones((3, 3), np.uint8)
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            noise_img = cv2.morphologyEx(noise_img, cv2.MORPH_CLOSE, kernel)
            noise_img = cv2.erode(noise_img, kernel, iterations=1)

            synthesized_base[n] = [noise_img, pattern_type]

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

        return pd.DataFrame(synthesized_base, columns=['waferMap', 'failureType'])

    def generator_random(self, plot=False):
        """

        :param plot:
        :return:
        """
        print('[INFO] Create random')
        # число синтезированных карт
        N_POINTS = self.number_points

        df_random = self.create_random(N_POINTS, pattern_type='Random', lam_poisson=2.1)

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, id in enumerate(sample_idx):
                ax[i].imshow(df_random.waferMap.values[id], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_random

    def create_synthesized_database(self, synthesized_path_name):
        """

        :param synthesized_path_name:
        :return:
        """

        df_scratch_curved = self.generator_scratch(mode=0)
        df_donut = self.generator_rings(mode=0)
        df_loc = self.generator_loc(mode=0)
        df_center = self.generator_center(mode=0)
        df_edge_ring = self.generator_edge_ring(mode=0)
        df_edge_loc = self.generator_edge_loc(mode=0)
        df_near_full = self.generator_near_full()
        df_random = self.generator_random()

        df = pd.concat([df_center, df_donut, df_edge_loc,
                        df_edge_ring, df_loc, df_near_full,
                        df_random, df_scratch_curved], sort=False)

        mapping_type = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2,
                        'Edge-Ring': 3, 'Loc': 4, 'Random': 5,
                        'Scratch': 6, 'Near-full': 7}

        df['failureNum'] = df.failureType
        df = df.replace({'failureNum': mapping_type})

        df.to_pickle('input/' + synthesized_path_name)

        return True


class TrainingDatabaseCreator(object):
    """

    """
    def __init__(self):
        self.full_database_path = 'input/LSWMD.pkl'
        self.database_only_patterns_path = 'input/df_withpattern.pkl'
        self.IMAGE_DIMS = (96, 96, 1)

    def read_full_data(self, synthesized_path_name=None):
        """

        :param synthesized_path_name:
        :return:
        """
        print('[INFO] Reading databases...')
        start_time = time.time()
        try:
            full_real_database = pd.read_pickle(self.database_only_patterns_path)
        except FileNotFoundError:
            print('[INFO] Prepared full database not found\n'
                  'Loading full database...')
            full_real_database = pd.read_pickle(self.full_database_path)
            mapping_type = {'Center': 0, 'Donut': 1, 'Edge-Loc': 2,
                            'Edge-Ring': 3, 'Loc': 4, 'Random': 5,
                            'Scratch': 6, 'Near-full': 7, 'none': 8}
            full_real_database['failureNum'] = full_real_database.failureType
            full_real_database = full_real_database.replace({'failureNum': mapping_type})
            full_real_database = full_real_database[(full_real_database['failureNum'] >= 0) &
                                                    (full_real_database['failureNum'] <= 7)]
            full_real_database = full_real_database.reset_index()
            full_real_database = full_real_database.drop(labels=['dieSize', 'lotName', 'waferIndex',
                                                                 'trianTestLabel', 'index'], axis=1)
            full_real_database.to_pickle(self.database_only_patterns_path)

        synthesized_database = None
        if synthesized_path_name:
            synthesized_database = pd.read_pickle('input/' + synthesized_path_name)
        else:
            print('[INFO] Synthesized database not found')

        print('reserved time: {:.2f}s'.format(time.time() - start_time))
        return full_real_database, synthesized_database

    def make_training_database(self, synthesized_path_name, failure_types_ratio):

        full_real_database, synthesized_database = self.read_full_data(synthesized_path_name)
        print('[INFO] Making train/test/val databases...')
        start_time = time.time()
        try:
            synthesized_database['failureType'] = synthesized_database['failureType'].map(lambda label: [[label]])
        except TypeError:
            print('Please, enter a path of the synthesized database')
            return None
        full_real_database['waferMap'] = full_real_database['waferMap'].map(lambda waf_map:
                                                                            cv2.resize(waf_map,
                                                                                       dsize=(self.IMAGE_DIMS[0],
                                                                                              self.IMAGE_DIMS[1]),
                                                                                       interpolation=cv2.INTER_CUBIC))
        training_database = synthesized_database
        testing_database = None
        for failure_type in failure_types_ratio:
            train_real, test_real = train_test_split(full_real_database[full_real_database.failureType == failure_type],
                                                     test_size=failure_types_ratio[failure_type],
                                                     random_state=42,
                                                     shuffle=True)

            training_database = pd.concat([training_database, train_real], sort=False)
            try:
                testing_database = pd.concat([testing_database, test_real], sort=False)
            except ValueError:
                testing_database = test_real

        testing_database, val_database = train_test_split(testing_database,
                                                          test_size=0.3,
                                                          random_state=42,
                                                          shuffle=True)

        full_dim = full_real_database.shape[0] + synthesized_database.shape[0]
        print('Database Dimensions\n'
              'full database: {}\n'
              'train: {} - {:.4f}%\n'
              'test: {} - {:.4f}%\n'
              'val {} - {:.4f}%'.format(full_dim,
                                        training_database.shape, training_database.shape[0] / full_dim,
                                        testing_database.shape, testing_database.shape[0] / full_dim,
                                        val_database.shape, val_database.shape[0] / full_dim))
        print('reserved time: {:.2f}s'.format(time.time() - start_time))
        gc.collect()
        return training_database, testing_database, val_database


# create_data = SynthesizedDatabaseCreator()
# create_data.create_synthesized_database('synthesized_test_database.pkl')

# args = {'synthesized_path_name': 'synthesized_test_database.pkl',
#         'failure_types_ratio': {'Center': 0.1,
#                                 'Donut': 0.1,
#                                 'Edge-Loc': 0.1,
#                                 'Edge-Ring': 0.1,
#                                 'Loc': 0.1,
#                                 'Random': 0.1,
#                                 'Scratch': 0.1,
#                                 'Near-full': 0.1}
#         }
# data = TrainingDatabaseCreator()
# data.make_training_database(**args)
