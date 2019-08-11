import numpy as np
import pandas as pd
import os


def read_data(path):
    df_all_standart = pd.read_pickle(os.path.join(os.getcwd(), path))

    return df_all_standart

read_data(path='input\LSWMD.pkl')