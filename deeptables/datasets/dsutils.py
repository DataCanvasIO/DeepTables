# -*- coding:utf-8 -*-
import pandas as pd
import os
from deeptables.utils import dt_logging
from sklearn import datasets
logger = dt_logging.get_logger(__name__)
basedir = os.path.dirname(__file__)


def load_boston():
    boston_dataset = datasets.load_boston()
    data = pd.DataFrame(boston_dataset.data)
    data.columns = boston_dataset.feature_names
    data.insert(0, 'target', boston_dataset.target)
    return data


def load_heart_disease_uci():
    print(f'Base dir:{basedir}')
    data = pd.read_csv(f'{basedir}/heart-disease-uci.csv')
    logger.info(f'data shape:{data.shape}')
    return data


def load_adult():
    print(f'Base dir:{basedir}')
    data = pd.read_csv(f'{basedir}/adult-uci.csv', header=None)
    logger.info(f'data shape:{data.shape}')
    return data


def load_glass_uci():
    print(f'Base dir:{basedir}')
    data = pd.read_csv(f'{basedir}/glass_uci.csv', header=None)
    logger.info(f'data shape:{data.shape}')
    return data


def load_bank():
    logger.info(f'Base dir:{os.getcwd()}')
    data = pd.read_csv(f'{basedir}/bank-uci.csv')
    logger.info(f'data shape:{data.shape}')
    return data
