# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
from ..utils import dt_logging, consts

logger = dt_logging.get_logger()


class LgbmLeavesEncoder:
    def __init__(self, cat_vars, cont_vars, task, **params):
        self.lgbm = None
        self.cat_vars = cat_vars
        self.cont_vars = cont_vars
        self.new_columns = []
        self.task = task
        self.lgbm_params = params

    def fit(self, X, y):
        from lightgbm import LGBMClassifier, LGBMRegressor

        X[self.cont_vars] = X[self.cont_vars].astype('float')
        X[self.cat_vars] = X[self.cat_vars].astype('int')
        logger.info(f'LightGBM task:{self.task}')
        if self.task == consts.TASK_MULTICLASS:  # multiclass label
            num_class = y.shape[-1]
            if self.lgbm_params is None:
                self.lgbm_params = {}
            self.lgbm_params['num_class'] = num_class
            self.lgbm_params['n_estimators'] = int(100 / num_class) + 1

            y = y.argmax(axis=-1)
        if self.task == consts.TASK_REGRESSION:
            self.lgbm = LGBMRegressor(**self.lgbm_params)
        else:
            self.lgbm = LGBMClassifier(**self.lgbm_params)
        self.lgbm.fit(X, y)

    def transform(self, X):
        X[self.cont_vars] = X[self.cont_vars].astype('float')
        X[self.cat_vars] = X[self.cat_vars].astype('int')

        leaves = self.lgbm.predict(X, pred_leaf=True, num_iteration=self.lgbm.best_iteration_)
        df_leaves = pd.DataFrame(leaves)
        self.new_columns = ['lgbm_leaf_' + str(i) for i in range(leaves.shape[1])]
        df_leaves.columns = self.new_columns
        return pd.concat([X, df_leaves], axis=1)

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


class CategorizeEncoder:
    def __init__(self, columns=None, remain_numeric=True):
        self.columns = columns
        self.remain_numeric = remain_numeric
        self.new_columns = []
        self.encoders = {}

    def fit(self, X):
        if self.columns is None:
            self.columns = X.columns.tolist()
        return self

    def transform(self, X):
        self.new_columns = []
        for col in self.columns:
            if self.remain_numeric:
                target_col = col + consts.COLUMNNAME_POSTFIX_CATEGORIZE
                self.new_columns.append((target_col, 'str', X[col].nunique()))
            else:
                target_col = col
            X[target_col] = X[col].astype('str')
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class MultiLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns
        self.encoders = {}

    def fit(self, X):
        if self.columns is None:
            self.columns = X.columns.tolist()
        for col in self.columns:
            logger.debug(f'LabelEncoder fitting [{col}]')
            if X.loc[:, col].dtype == 'object':
                X.loc[:, col] = X.loc[:, col].astype('str')
                # print(f'Column "{col}" has been convert to "str" type.')
            le = SafeLabelEncoder()
            le.fit(X.loc[:, col])
            self.encoders[col] = le
        return self

    def transform(self, X):
        for col in self.columns:
            logger.debug(f'LabelEncoder transform [{col}]')
            if X.loc[:, col].dtype == 'object':
                X.loc[:, col] = X.loc[:, col].astype('str')
                # print(f'Column "{col}" has been convert to "str" type.')
            X.loc[:, col] = self.encoders[col].transform(X.loc[:, col])
        return X

    def fit_transform(self, X):
        if self.columns is None:
            self.columns = X.columns.tolist()
        for col in self.columns:
            logger.debug(f'LabelEncoder fit_transform [{col}]')
            if X.loc[:, col].dtype == 'object':
                X.loc[:, col] = X.loc[:, col].astype('str')
                # print(f'Column "{col}" has been convert to "str" type.')
            le = SafeLabelEncoder()
            X.loc[:, col] = le.fit_transform(X.loc[:, col])
            self.encoders[col] = le
        return X


class MultiKBinsDiscretizer:

    def __init__(self, columns=None, bins=None, strategy='quantile'):
        logger.info(f'{len(columns)} variables to discrete.')
        self.columns = columns
        self.bins = bins
        self.stragegy = strategy
        self.new_columns = []
        self.encoders = {}

    def fit(self, X):
        self.new_columns = []
        if self.columns is None:
            self.columns = X.columns.tolist()
        for col in self.columns:
            new_name = col + consts.COLUMNNAME_POSTFIX_DISCRETE
            n_unique = X.loc[:, col].nunique()
            n_null = X.loc[:, col].isnull().sum()
            c_bins = self.bins
            if c_bins is None or c_bins <= 0:
                c_bins = round(n_unique ** 0.25) + 1
            encoder = KBinsDiscretizer(n_bins=c_bins, encode='ordinal', strategy=self.stragegy)
            self.new_columns.append((col, new_name, encoder.n_bins))
            encoder.fit(X[[col]])
            self.encoders[col] = encoder
        return self

    def transform(self, X):
        for col in self.columns:
            new_name = col + consts.COLUMNNAME_POSTFIX_DISCRETE
            encoder = self.encoders[col]
            nc = encoder.transform(X[[col]]).astype(consts.DATATYPE_LABEL).reshape(-1)
            X[new_name] = nc
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class DataFrameWrapper:
    def __init__(self, transform, columns=None):
        self.transformer = transform
        self.columns = columns

    def fit(self, X):
        if self.columns is None:
            self.columns = X.columns.tolist()
        self.transformer.fit(X)
        return self

    def transform(self, X):
        df = pd.DataFrame(self.transformer.transform(X))
        df.columns = self.columns
        return df

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


class SafeLabelEncoder(LabelEncoder):
    def transform(self, y):
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        unseen = len(self.classes_)
        y = np.array([np.searchsorted(self.classes_, x) if x in self.classes_ else unseen for x in y])
        return y


class GaussRankScaler:
    def __init__(self):
        self.epsilon = 0.001
        self.lower = -1 + self.epsilon
        self.upper = 1 - self.epsilon
        self.range = self.upper - self.lower
        self.divider = None

    def fit_transform(self, X):
        from scipy.special import erfinv
        i = np.argsort(X, axis=0)
        j = np.argsort(i, axis=0)

        assert (j.min() == 0).all()
        assert (j.max() == len(j) - 1).all()

        j_range = len(j) - 1
        self.divider = j_range / self.range

        transformed = j / self.divider
        transformed = transformed - self.upper
        transformed = erfinv(transformed)

        return transformed
