import os
import gc
import time
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from src.utils.transform import *

import warnings
warnings.filterwarnings("ignore")


class SynthesizedDatabaseCreator(object):
    """

    """
    def __init__(self, example_number, synthesized_path_name, image_dims):
        self.IMAGE_DIMS = image_dims
        self.number_points = example_number
        self.cpu_count = 4
        self.synthesized_path_name = synthesized_path_name

        def add_pattern(cur_x, add_x):
            if cur_x == 1 and add_x == 2:
                return add_x
            else:
                return cur_x
        self.add_pattern = np.vectorize(add_pattern)

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

    def sawtooth_line(self, XC_, YC_, L0_, angle_, pattern_type, line_count=1, lam_poisson=0.2, save=False,
                      add_patterns=[None]):
        size = XC_.shape[0]
        synthesized_base = [None] * size

        for n in tqdm(range(size)):
            step = n
            template = deepcopy(self.template_map)

            if add_patterns[0]:
                for pattern in add_patterns:
                    for img_pattern in pattern:
                        template = self.add_pattern(template, img_pattern)

            COLOR_SCALE = 2
            for repeate in range(line_count):
                if repeate:
                    step = random.randint(0, size - 1)
                # иниицализация параметров прямой
                L0 = L0_[step]
                XC = XC_[step]
                YC = YC_[step]
                angle = angle_[step]

                # параметры уравнения
                def delta_(x, y):
                    return int(math.sqrt(x ** 2 + y ** 2))

                delta = np.vectorize(delta_)
                L = L0 - np.sum(delta(XC, YC)[1:])
                N = 200
                x0, y0 = 0, 0

                # кусочное построение пилообразной прямой
                for i in range(XC.shape[0]):
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

            synthesized_base[n] = [template, pattern_type]

            # для презентации
            if save:
                path = 'output/test_classes/{}'.format(pattern_type)
                try:
                    os.mkdir(path)
                except OSError:
                    pass
                plt.imshow(template, cmap='inferno')
                name = '/{}{}.jpg'.format(pattern_type, n)
                plt.savefig(path + name)

        return pd.DataFrame(synthesized_base, columns=['waferMap', 'failureType'])

    @staticmethod
    def add_noise(template, pattern_type, lam_poisson=0.2, dilate_time=1):
        # расширение по соседу
        is_dilate = random.randint(-1, 1)
        if is_dilate == 1 or pattern_type == 'scratch':
            kernel1 = np.ones((3, 3), np.uint8)
            kernel2 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            count_iter = random.randint(1, dilate_time)
            template = cv2.dilate(template, kernel2, iterations=count_iter)
            template = cv2.morphologyEx(template, cv2.MORPH_CLOSE, kernel2)

        # внесем шум
        noise_img = template.copy()
        mask = np.random.randint(0, 2, size=noise_img.shape).astype(np.bool)
        mask[noise_img == 0] = False
        r = np.random.poisson(lam=lam_poisson, size=noise_img.shape)
        # нормировка на величину шума
        r[r == 0] = 1
        r[r > 2] = 2
        noise_img[mask] = r[mask]

        # расширение
        # kernel = np.ones((3, 3), np.uint8)
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        noise_img = cv2.morphologyEx(noise_img, cv2.MORPH_CLOSE, kernel)

    #     if pattern_type != 'scratch':
    #         noise_img = cv2.erode(noise_img, kernel, iterations=1)

        return noise_img

    def generator_scratch(self, mode=0, plot=False, line_count=1, add_patterns=[None], is_noised=False):
        print('[INFO] Create scratches')
        # число синтезированных карт
        N_POINTS = self.number_points // 2
        line_part = 5  # сегментов в одной линии

        # суммарная длина отрезка
        L0 = np.random.randint(0.2 * self.IMAGE_DIMS[0], 0.45 * self.IMAGE_DIMS[0], size=N_POINTS)

        # X координата старта прямой
        xc = [np.random.randint(0.2 * self.IMAGE_DIMS[0], 0.5 * self.IMAGE_DIMS[0], size=N_POINTS)]
        for _ in range(line_part - 1):
            # смещение по x для старта следующей прямой
            delta_xc = np.random.randint(0.01 * self.IMAGE_DIMS[0], 0.02 * self.IMAGE_DIMS[0] + 2, size=N_POINTS)
            np.random.shuffle(delta_xc)
            xc.append(delta_xc)
        # merge под формат генератора
        xc = np.array(xc).T
        np.random.shuffle(xc)

        # Y координата старта прямой
        yc = [np.random.randint(0.3 * self.IMAGE_DIMS[0], 0.7 * self.IMAGE_DIMS[0], size=N_POINTS)]
        for _ in range(line_part - 1):
            # смещение по x для старта следующей прямой
            delta_yc = np.random.randint(0.01 * self.IMAGE_DIMS[0], 0.02 * self.IMAGE_DIMS[0] + 2, size=N_POINTS)
            np.random.shuffle(delta_yc)
            yc.append(delta_yc)
        # merge под формат генератора
        yc = np.array(yc).T
        np.random.shuffle(yc)

        # углы наклона для каждого отрезка
        angle = [np.random.randint(-50, 50, size=N_POINTS) * np.pi / 180]
        for _ in range(line_part - 1):
            part_angle = np.random.randint(30, 40, size=N_POINTS) * np.pi / 180 * np.sign(angle[0])
            angle.append(part_angle)
        angle = np.array(angle).T
        np.random.shuffle(angle)

        df_scratch_curved = None
        if mode == 0:
            # генератор параллельный
            n_workers = self.cpu_count
            results = Parallel(n_workers)(
                delayed(self.sawtooth_line)(xc[i::n_workers], yc[i::n_workers],
                                       L0[i::n_workers], angle[i::n_workers],
                                       pattern_type='Scratch',
                                       line_count=line_count,
                                       add_patterns=add_patterns)
                for i in range(n_workers))

            df_scratch_curved = results[0]
            for i in range(1, len(results)):
                df_scratch_curved = pd.concat((df_scratch_curved, results[i]), sort=False)

        if is_noised:
            df_scratch_curved.waferMap = df_scratch_curved.waferMap.map(lambda wafer_map:
                                                                        self.add_noise(wafer_map,
                                                                                       pattern_type='scratch',
                                                                                       lam_poisson=0.3))

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(15, 10))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, idx in enumerate(sample_idx):
                ax[i].imshow(df_scratch_curved.waferMap.values[idx], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_scratch_curved

    def create_rings(self, XC, YC, R_, PHI, N, pattern_type, lam_poisson=1.2, save=False, add_patterns=[None]):
        color_scale = 2
        size = XC.shape[0]
        synthesized_base = [None] * size

        for n in tqdm(range(size)):
            # тестовый полигон
            template = deepcopy(self.template_map)

            if add_patterns[0]:
                for pattern in add_patterns:
                    for img_pattern in pattern:
                        template = self.add_pattern(template, img_pattern)

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

            for idx in points:
                i, j = idx
                i = int(round(i))
                j = int(round(j))
                try:
                    if template[i, j] == 1:
                        template[i, j] = color_scale
                except IndexError:
                    break

            synthesized_base[n] = [template, pattern_type]

            # для презентации
            if save:
                path = 'output/test_classes/{}'.format(pattern_type)
                try:
                    os.mkdir(path)
                except OSError:
                    pass
                plt.imshow(template, cmap='inferno')
                name = '/{}{}.jpg'.format(pattern_type, n)
                plt.savefig(path + name)

        return pd.DataFrame(synthesized_base, columns=['waferMap', 'failureType'])

    def generator_donut(self, mode=0, plot=False, add_patterns=None, is_noised=False):
        print('[INFO] Create donuts')
        # число синтезированных карт
        N_POINTS = self.number_points

        PHI = None
        for i in range(4):
            # угол старта для сектора
            phi1 = np.random.uniform(0 + 95 * i, 30 + 95 * i, size=N_POINTS // 4) * np.pi / 180
            # угол конца для сектора
            phi2 = np.random.uniform(180 + 90 * i, 360 * (i + 1), size=N_POINTS // 4) * np.pi / 180
            phi = np.vstack((phi1, phi2))
            # merge под формат генератора
            phi = np.array([[phi[0, j], phi[1, j]] for j in range(phi.shape[1])])
            if i == 0:
                PHI = phi
            else:
                PHI = np.vstack((PHI, phi))

        # радиус внутреннего круга
        r1 = np.random.randint(0.15 * self.IMAGE_DIMS[0], 0.3 * self.IMAGE_DIMS[0], size=N_POINTS)
        # радиус внешнего круга
        r2 = np.random.randint(0.33 * self.IMAGE_DIMS[0], 0.4 * self.IMAGE_DIMS[0], size=N_POINTS)
        r = np.vstack((r1, r2))
        # merge под формат генератора
        r = np.array([[r[0, i], r[1, i]] for i in range(r.shape[1])])

        # X координата старта прямой
        XC = np.random.randint(0.45 * self.IMAGE_DIMS[0], 0.55 * self.IMAGE_DIMS[0], size=N_POINTS)
        # Y координата старта прямой
        YC = np.random.randint(0.45 * self.IMAGE_DIMS[0], 0.55 * self.IMAGE_DIMS[0], size=N_POINTS)

        # интесивность
        N = np.random.randint(200, 210, size=N_POINTS)

        df_donut = None
        if mode == 0:
            # генератор параллельный
            n_workers = self.cpu_count
            results = Parallel(n_workers)(
                delayed(self.create_rings)(XC[i::n_workers], YC[i::n_workers],
                                      r[i::n_workers], PHI[i::n_workers],
                                      N[i::n_workers], pattern_type='Donut',
                                      add_patterns=add_patterns)
                for i in range(n_workers))

            df_donut = results[0]
            for i in range(1, len(results)):
                df_donut = pd.concat((df_donut, results[i]), sort=False)

        if is_noised:
            df_donut.waferMap = df_donut.waferMap.map(lambda wafer_map:
                                                      self.add_noise(wafer_map,
                                                                     pattern_type='donut',
                                                                     lam_poisson=0.9,
                                                                     dilate_time=4))

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, idx in enumerate(sample_idx):
                ax[i].imshow(df_donut.waferMap.values[idx], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_donut

    def generator_loc(self, mode=0, plot=False, add_patterns=[None], is_noised=True):
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
                                           N[i::n_workers], pattern_type='Loc',
                                           add_patterns=add_patterns)
                for i in range(n_workers))

            df_loc = results[0]
            for i in range(1, len(results)):
                df_loc = pd.concat((df_loc, results[i]), sort=False)

        if is_noised:
            df_loc.waferMap = df_loc.waferMap.map(lambda wafer_map:
                                                  self.add_noise(wafer_map,
                                                                 pattern_type='scratch',
                                                                 lam_poisson=0.3))

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, idx in enumerate(sample_idx):
                ax[i].imshow(df_loc.waferMap.values[idx], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_loc

    def generator_center(self, mode=0, plot=False, add_patterns=[None], is_noised=True):
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
                                           N[i::n_workers], pattern_type='Center',
                                           add_patterns=add_patterns)
                for i in range(n_workers))

            df_center = results[0]
            for i in range(1, len(results)):
                df_center = pd.concat((df_center, results[i]), sort=False)

        if is_noised:
            df_center.waferMap = df_center.waferMap.map(lambda wafer_map:
                                                        self.add_noise(wafer_map,
                                                                       pattern_type='scratch',
                                                                       lam_poisson=0.3))

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, idx in enumerate(sample_idx):
                ax[i].imshow(df_center.waferMap.values[idx], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_center

    def generator_edge_ring(self, mode=0, plot=False, add_patterns=[None], is_noised=True):
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
                                           N[i::n_workers], pattern_type='Edge-Ring',
                                           add_patterns=add_patterns)
                for i in range(n_workers))

            df_edge_ring = results[0]
            for i in range(1, len(results)):
                df_edge_ring = pd.concat((df_edge_ring, results[i]), sort=False)

        if is_noised:
            df_edge_ring.waferMap = df_edge_ring.waferMap.map(lambda wafer_map:
                                                              self.add_noise(wafer_map,
                                                                             pattern_type='scratch',
                                                                             lam_poisson=0.3))

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, idx in enumerate(sample_idx):
                ax[i].imshow(df_edge_ring.waferMap.values[idx], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_edge_ring

    def generator_edge_loc(self, mode=0, plot=False, add_patterns=[None], is_noised=True):
        print('[INFO] Create edge_loc')
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
                                           N[i::n_workers], pattern_type='Edge-Loc',
                                           add_patterns=add_patterns)
                for i in range(n_workers))

            df_edge_loc = results[0]
            for i in range(1, len(results)):
                df_edge_loc = pd.concat((df_edge_loc, results[i]), sort=False)

        if is_noised:
            df_edge_loc.waferMap = df_edge_loc.waferMap.map(lambda wafer_map:
                                                            self.add_noise(wafer_map,
                                                                           pattern_type='scratch',
                                                                           lam_poisson=0.3))

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, idx in enumerate(sample_idx):
                ax[i].imshow(df_edge_loc.waferMap.values[idx], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_edge_loc

    def create_near_full(self, capacity, pattern_type, lam_poisson=1.2, save=False):
        synthesized_base = [None] * capacity
        for step in range(capacity):
            # тестовый полигон
            template = deepcopy(self.template_map)

            # внесем шум
            noise_img = deepcopy(template)
            mask = np.random.randint(0, 2, size=noise_img.shape).astype(np.bool)
            mask[noise_img == 0] = False
            r = np.random.poisson(lam=lam_poisson, size=noise_img.shape)
            # нормировка на шумы
            # r = np.around(r//np.max(r))
            r[r == 0] = 1
            r[r == 1] = 2
            r[r > 2] = 1
            noise_img[mask] = r[mask]

            # сверткой расширим
            kernel = np.ones((3, 3), np.uint8)
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            noise_img = cv2.morphologyEx(noise_img, cv2.MORPH_CLOSE, kernel)
            noise_img = cv2.erode(noise_img, kernel, iterations=1)

            synthesized_base[step] = [noise_img, pattern_type]

            # для презенташки
            if save:
                path = 'output/test_classes/{}'.format(pattern_type)
                try:
                    os.mkdir(path)
                except OSError:
                    pass
                plt.imshow(noise_img, cmap='inferno')
                name = '/{}{}.jpg'.format(pattern_type, step)
                plt.savefig(path + name)

        return pd.DataFrame(synthesized_base, columns=['waferMap', 'failureType'])

    def generator_near_full(self, plot=False):
        print('[INFO] Create near_full')
        # число синтезированных карт
        N_POINTS = self.number_points

        df_near_full = self.create_near_full(N_POINTS, pattern_type='Near-full', lam_poisson=1.3)

        if plot:
            fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10, 8))
            ax = ax.ravel(order='C')
            sample_idx = np.random.choice(N_POINTS, 100)
            for i, idx in enumerate(sample_idx):
                ax[i].imshow(df_near_full.waferMap.values[idx], cmap='inferno')
                ax[i].axis('off')
            fig.suptitle('Synthesized scratches')
            plt.show()
        else:
            gc.collect()

        return df_near_full

    def create_random(self, capacity, pattern_type, lam_poisson=1.2, save=False):
        synthesized_base = [None] * capacity
        for step in tqdm(range(capacity)):
            # тестовый полигон
            template = deepcopy(self.template_map)

            # внесем шум
            noise_img = deepcopy(template)
            mask = np.random.randint(0, 2, size=noise_img.shape).astype(np.bool)
            mask[noise_img == 0] = False
            r = np.random.poisson(lam=lam_poisson, size=noise_img.shape)
            # нормировка на шумы
            # r = np.around(r//np.max(r))
            r[r == 0] = 1
            r[r > 2] = 2
            noise_img[mask] = r[mask]

            # сверткой расширим
            kernel = np.ones((3, 3), np.uint8)
            kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
            noise_img = cv2.morphologyEx(noise_img, cv2.MORPH_CLOSE, kernel)
            noise_img = cv2.erode(noise_img, kernel, iterations=1)

            synthesized_base[step] = [noise_img, pattern_type]

            # для презенташки
            if save:
                path = 'output/test_classes/{}'.format(pattern_type)
                try:
                    os.mkdir(path)
                except OSError:
                    pass
                plt.imshow(noise_img, cmap='inferno')
                name = '/{}{}.jpg'.format(pattern_type, step)
                plt.savefig(path + name)

        return pd.DataFrame(synthesized_base, columns=['waferMap', 'failureType'])

    def generator_random(self, plot=False):
        print('[INFO] Create random')
        # число синтезированных карт
        N_POINTS = self.number_points

        df_random = self.create_random(N_POINTS, pattern_type='none', lam_poisson=2.1)

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

    def create_synthesized_database(self, classes, is_noised):
        df_scratch_curved = [self.generator_scratch(mode=0, plot=False, line_count=i+1,
                                                    add_patterns=[None], is_noised=is_noised)
                             for i in range(2)]
        df_scratch = pd.concat(df_scratch_curved, ignore_index=True)

        df_donut = self.generator_donut(mode=0, plot=False, add_patterns=[None], is_noised=is_noised)
        df_loc = self.generator_loc(mode=0, plot=False, add_patterns=[None], is_noised=is_noised)
        df_center = self.generator_center(mode=0, plot=False, add_patterns=[None], is_noised=is_noised)
        df_edge_ring = self.generator_edge_ring(mode=0, plot=False, add_patterns=[None], is_noised=is_noised)
        df_edge_loc = self.generator_edge_loc(mode=0, plot=False, add_patterns=[None], is_noised=is_noised)
        df_random = self.generator_random(plot=False)


        data = [df_center, df_donut, df_loc,
                df_scratch, df_edge_ring, df_edge_loc,
                df_random
                ]
        df = pd.concat(data[:classes], sort=False)

        mapping_type = {'Center': 0, 'Donut': 1, 'Loc': 2,
                        'Scratch': 3, 'Edge-Ring': 4, 'Edge-Loc': 5,
                        'none': 6
                        }
        mapping_type = dict(list(iter(mapping_type.items()))[:classes])

        df['failureNum'] = df.failureType
        df = df.replace({'failureNum': mapping_type})

        df.to_pickle('input/' + self.synthesized_path_name)

        return True


class TrainingDatabaseCreator(object):
    """

    """
    def __init__(self, database_only_patterns_path):
        self.full_database_path = 'input/LSWMD.pkl'
        self.database_only_patterns_path = 'input/' + database_only_patterns_path
        self.IMAGE_DIMS = (1, 96, 96)

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
                  'Loading full database...'
                  f'Create {self.database_only_patterns_path} database')

            full_real_database = pd.read_pickle(self.full_database_path)
            mapping_type = {'Center': 0, 'Donut': 1, 'Loc': 2,
                            'Scratch': 3, 'Edge-Ring': 4,  'Edge-Loc': 5,
                            'none': 6, 'Near-full': 7, 'Random': 8}
            full_real_database['failureNum'] = full_real_database.failureType
            full_real_database = full_real_database.replace({'failureNum': mapping_type})
            full_real_database = full_real_database[(full_real_database['failureNum'] >= 0) &
                                                    (full_real_database['failureNum'] <= 5)]
            full_real_database = full_real_database.reset_index()
            full_real_database = full_real_database.drop(labels=['dieSize', 'lotName', 'waferIndex',
                                                                 'trianTestLabel', 'index'], axis=1)

            ######################
            # Get fixed size of maps
            out_map = []
            out_class = []
            dim_size = 40
            for index, row in full_real_database.iterrows():
                waf_map = row.waferMap
                waf_type = row.failureType

                if waf_map.shape[0] > dim_size and waf_map.shape[1] > dim_size:
                    out_map += [waf_map]
                    out_class += [waf_type[0][0]]

            database = pd.DataFrame(data=np.vstack((out_map, out_class)).T, columns=['waferMap', 'failureType'])
            database['failureNum'] = database.failureType
            database = database.replace({'failureNum': mapping_type})

            full_real_database = database
            database.to_pickle(self.database_only_patterns_path)

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
            synthesized_database['failureType'] = synthesized_database['failureType'].map(lambda label: label)
        except TypeError:
            print('Please, enter a path of the synthesized database')
            return None
        full_real_database['waferMap'] = full_real_database['waferMap'].map(lambda waf_map:
                                                                            cv2.resize(waf_map,
                                                                                       dsize=(self.IMAGE_DIMS[1],
                                                                                              self.IMAGE_DIMS[1]),
                                                                                       interpolation=cv2.INTER_NEAREST))
        training_database = synthesized_database
        testing_database = None
        for failure_type in failure_types_ratio:
            train_real, test_real = train_test_split(full_real_database[full_real_database.failureType == failure_type],
                                                     train_size=failure_types_ratio[failure_type],
                                                     random_state=42,
                                                     shuffle=True)

            training_database = pd.concat([training_database, train_real], sort=False)
            try:
                testing_database = pd.concat([testing_database, test_real], sort=False)
            except ValueError:
                testing_database = test_real

        testing_database, val_database = train_test_split(testing_database,
                                                          test_size=0.7,
                                                          random_state=42,
                                                          shuffle=True)

        full_dim = full_real_database.shape[0] + synthesized_database.shape[0]
        print('Database Dimensions\n'
              '* full database: {}, real: {}, synth: {}\n'
              '* train: {} - {:.4f}%\n'
              '* test: {} - {:.4f}%\n'
              '* val {} - {:.4f}%'.format(full_dim, full_real_database.shape[0], synthesized_database.shape[0],
                                          training_database.shape, training_database.shape[0] / full_dim,
                                          testing_database.shape, testing_database.shape[0] / full_dim,
                                          val_database.shape, val_database.shape[0] / full_dim))

        uni = np.unique(training_database.failureNum, return_counts=True)[1]
        print(f'[TRAIN] Center: {uni[0]}, Donut: {uni[1]}, Loc: {uni[2]}, Scratch: {uni[3]}, Edge-Ring: {uni[4]}, Edge-Loc: {uni[5]}, none: {uni[6]}')

        uni = np.unique(val_database.failureNum, return_counts=True)[1]
        print(f'[VAL] Center: {uni[0]}, Donut: {uni[1]}, Loc: {uni[2]}, Scratch: {uni[3]}, Edge-Ring: {uni[4]}, Edge-Loc: {uni[5]}, none: {uni[6]}')

        uni = np.unique(testing_database.failureNum, return_counts=True)[1]
        print(f'[TEST] Center: {uni[0]}, Donut: {uni[1]}, Loc: {uni[2]}, Scratch: {uni[3]}, Edge-Ring: {uni[4]}, Edge-Loc: {uni[5]}, none: {uni[6]}')

        print('reserved time: {:.2f}s'.format(time.time() - start_time))

        gc.collect()
        return training_database, testing_database, val_database

    def get_fixed_size_dataset(self, synthesized_path_name, num_each_pattern):
        full_real_database, synthesized_database = self.read_full_data(synthesized_path_name)

        train_idx = []
        for pattern_id in range(len(np.unique(full_real_database['failureNum']))):
            pattern_indexes = full_real_database.index[full_real_database['failureNum'] == pattern_id].tolist()
            sample_idx = np.random.choice(pattern_indexes, num_each_pattern, replace=False)
            train_idx.append(sample_idx)

        train_idx = np.array(train_idx).flatten()
        training_database = full_real_database.iloc[train_idx]

        val_idx = np.random.choice(np.arange(training_database.shape[0]),
                                   int(0.4 * training_database.shape[0]),
                                   replace=False)
        val_database = training_database.iloc[val_idx].sample(frac=1).reset_index(drop=True)

        training_database = training_database.iloc[np.delete(np.arange(training_database.shape[0]),
                                                             val_idx)].sample(frac=1).reset_index(drop=True)

        training_database = pd.concat((synthesized_database,
                                       training_database)).sample(frac=1).reset_index(drop=True)

        testing_database = full_real_database.iloc[np.delete(np.arange(full_real_database.shape[0]),
                                                             train_idx)].sample(frac=1).reset_index(drop=True)
        gc.collect()
        return training_database, testing_database, val_database


class WaferDataset(Dataset):
    def __init__(self, image_list, mode, label_list=None,
                 fine_size=96, pad_left=0, pad_right=0):
        self.image_list = image_list
        self.mode = mode
        self.label_list = label_list
        self.fine_size = fine_size
        self.pad_left = pad_left
        self.pad_right = pad_right

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image = deepcopy(self.image_list[idx])

        if self.mode == 'train':
            label = deepcopy(self.label_list[idx])

            if self.fine_size != image.shape[0]:
                image = cv2.resize(image, dsize=(self.fine_size, self.fine_size))
            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)

            image = image.reshape(1, self.fine_size + self.pad_left + self.pad_right,
                                  self.fine_size + self.pad_left + self.pad_right)

            image, label = torch.from_numpy(image).float(), torch.tensor(label)
            return image, label

        elif self.mode == 'val':
            label = deepcopy(self.label_list[idx])

            if self.fine_size != image.shape[0]:
                image = cv2.resize(image, dsize=(self.fine_size, self.fine_size))
            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)

            image = image.reshape(1, self.fine_size + self.pad_left + self.pad_right,
                                  self.fine_size + self.pad_left + self.pad_right)

            image, label = torch.from_numpy(image).float(), torch.tensor(label)
            return image, label

        elif self.mode == 'test':
            if self.fine_size != image.shape[0]:
                image = cv2.resize(image, dsize=(self.fine_size, self.fine_size))
            if self.pad_left != 0:
                image = do_center_pad(image, self.pad_left, self.pad_right)

            image = image.reshape(1, self.fine_size + self.pad_left + self.pad_right,
                                  self.fine_size + self.pad_left + self.pad_right)

            image = torch.from_numpy(image).float()
            return image


if __name__ == '__main__':

    args = {'example_number': 500,
            'synthesized_path_name': 'synt_noise_c7_05k.pkl',
            'image_dims': (96, 96, 1)}

    create_data = SynthesizedDatabaseCreator(**args)
    create_data.create_synthesized_database(classes=7, is_noised=True)

    # data = TrainingDatabaseCreator('real_g40_c6.pkl')
    # train, test, val = data.get_fixed_size_dataset('synt_noise_c6_4k.pkl', 100)

    # args = {'synthesized_path_name': 'synt_noise_c7_v1.pkl',
    #         'failure_types_ratio': {'Center': 0.0,
    #                                 'Donut': 0.0,
    #                                 'Edge-Loc': 0.0,
    #                                 'Edge-Ring': 0.0,
    #                                 'Loc': 0.0,
    #                                 'Scratch': 0.0,
    #                                 'Random': 0.0}
    #         }
    # data = TrainingDatabaseCreator('real_g50_c7.pkl')
    # train, val, test = data.make_training_database(**args)
    #
    # train_list_im = list(train.waferMap.values)
    # train_label = list(train.failureNum.values)
    #
    # train_data = WaferDataset(list(train.waferMap.values), mode='train', label_list=list(train.failureNum.values))
    # train_loader = DataLoader(train_data,
    #                           shuffle=RandomSampler(train_data),
    #                           batch_size=50,
    #                           num_workers=1,
    #                           pin_memory=True)
    #
    # for image, label in train_loader:
    #     print(label)
