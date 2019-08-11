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


def read_data(path):
    df_all_standart = pd.read_pickle(os.path.join(os.getcwd(), path))

    return df_all_standart

read_data(path='input\LSWMD.pkl')