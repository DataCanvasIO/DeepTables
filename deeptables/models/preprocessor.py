# -*- coding:utf-8 -*-

import time
import collections
import numpy as np
import pandas as pd
import copy
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from .metainfo import CategoricalColumn, ContinuousColumn
from ..preprocessing import MultiLabelEncoder, MultiKBinsDiscretizer, DataFrameWrapper, LgbmLeavesEncoder, \
    CategorizeEncoder
from ..utils import dt_logging, consts
from . import deeptable

logger = dt_logging.get_logger()
from .config import ModelConfig


class AbstractPreprocessor:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.labels_ = None
        self.task_ = None

    @property
    def pos_label(self):
        if self.labels_ is not None and len(self.labels_) == 2:
            return self.labels_[1]
        else:
            return None

    @property
    def labels(self):
        return self.labels_

    @property
    def task(self):
        return self.task_

    def fit_transform(self, X, y, copy_data=True):
        raise NotImplementedError

    def transform_X(self, X, copy_data=True):
        raise NotImplementedError

    def transform_y(self, y, copy_data=True):
        raise NotImplementedError

    def transform(self, X, y, copy_data=True):
        raise NotImplementedError

    def inverse_transform_y(self, y_indicator):
        raise NotImplementedError

    def get_categorical_columns(self):
        raise NotImplementedError

    def get_continuous_columns(self):
        raise NotImplementedError

    def save(self, filepath):
        raise NotImplementedError

    @staticmethod
    def load(filepath):
        raise NotImplementedError


class DefaultPreprocessor(AbstractPreprocessor):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.reset()

    def reset(self):
        self.metainfo = None
        self.categorical_columns = None
        self.continuous_columns = None
        self.y_lable_encoder = None
        self.X_transformers = collections.OrderedDict()

    def prepare_X(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if len(set(X.columns)) != len(list(X.columns)):
            cols = [item for item, count in collections.Counter(X.columns).items() if count > 1]
            raise ValueError(f'Columns with duplicate names in X: {cols}')
        if X.columns.dtype != 'object':
            X.columns = ['x_' + str(c) for c in X.columns]
            logger.warn(f"Column index of X has been converted: {X.columns}")
        return X

    def fit_transform(self, X, y, copy_data=True):
        start = time.time()
        self.reset()
        if X is None:
            raise ValueError(f'X cannot be none.')
        if y is None:
            raise ValueError(f'y cannot be none.')
        if len(X.shape) != 2:
            raise ValueError(f'X must be a 2D datasets.')
        if len(y.shape) != 1:
            raise ValueError(f'y must be a 1D datasets.')
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"The number of samples of X and y must be the same. X.shape:{X.shape}, y.shape{y.shape}")
        if pd.Series(y).isnull().sum() > 0:
            raise ValueError("Missing values in y.")

        if copy:
            X = copy.deepcopy(X)
            y = copy.deepcopy(y)

        y = self.fit_transform_y(y)

        X = self.prepare_X(X)
        X = self.__prepare_features(X)
        if self.config.auto_imputation:
            X = self._imputation(X)
        if self.config.auto_encode_label:
            X = self._categorical_encoding(X)
        if self.config.auto_discrete:
            X = self._discretization(X)
        if self.config.apply_gbm_features and y is not None:
            X = self._apply_gbm_features(X, y)
        self.X_transformers['last'] = 'passthrough'

        print(f'fit_transform cost:{time.time() - start}')
        return X, y

    def fit_transform_y(self, y):
        self.task_, self.labels_ = deeptable.infer_task_type(y)
        if self.task_ in [consts.TASK_BINARY, consts.TASK_MULTICLASS]:
            self.y_lable_encoder = LabelEncoder()
            y = self.y_lable_encoder.fit_transform(y)
            self.labels_ = self.y_lable_encoder.classes_
        return y

    def transform(self, X, y, copy_data=True):
        X_t = self.transform_X(X, copy_data)
        y_t = self.transform_y(y, copy_data)
        return X_t, y_t

    def transform_y(self, y, copy_data=True):
        logger.info("Transform [y]...")
        start = time.time()
        if copy_data:
            y = copy.deepcopy(y)
        if self.y_lable_encoder is not None:
            y = self.y_lable_encoder.transform(y)
        print(f'transform_y cost:{time.time() - start}')
        y = np.array(y)
        return y

    def transform_X(self, X, copy_data=True):
        start = time.time()
        logger.info("Transform [X]...")
        if copy_data:
            X = copy.deepcopy(X)
        X = self.prepare_X(X)
        steps = (step for step in self.X_transformers.values())
        pipeline = make_pipeline(*steps)
        X_t = pipeline.transform(X)
        print(f'transform_X cost:{time.time() - start}')
        return X_t

    def inverse_transform_y(self, y_indicator):
        if self.y_lable_encoder is not None:
            return self.y_lable_encoder.inverse_transform(y_indicator)
        else:
            return y_indicator

    def __prepare_features(self, X):
        start = time.time()

        logger.info(f'Preparing features...')
        num_vars = []
        convert2cat_vars = []
        cat_vars = []
        excluded_vars = []

        if self.config.cat_exponent >= 1:
            raise ValueError(f'"cat_expoent" must be less than 1, not {self.config.cat_exponent} .')

        unique_upper_limit = round(X.shape[0] ** self.config.cat_exponent)
        for c in X.columns:
            nunique = X[c].nunique()
            dtype = str(X[c].dtype)

            if c in self.config.exclude_columns:
                excluded_vars.append((c, dtype, nunique))
                continue
            if self.config.categorical_columns is not None and isinstance(self.config.categorical_columns, list):
                if c in self.config.categorical_columns:
                    cat_vars.append((c, dtype, nunique))
                else:
                    if np.issubdtype(dtype, np.number):
                        num_vars.append((c, dtype, nunique))
                    else:
                        print(
                            f'Column [{c}] has been discarded. It is not numeric and not in [config.categorical_columns].')
            else:
                if dtype == 'object' or dtype == 'category' or dtype == 'bool':
                    cat_vars.append((c, dtype, nunique))
                elif self.config.auto_categorize and X[c].nunique() < unique_upper_limit:
                    convert2cat_vars.append((c, dtype, nunique))
                else:
                    num_vars.append((c, dtype, nunique))

        if len(convert2cat_vars) > 0:
            ce = CategorizeEncoder([c for c, d, n in convert2cat_vars], self.config.cat_remain_numeric)
            X = ce.fit_transform(X)
            self.X_transformers['categorize'] = ce
            if self.config.cat_remain_numeric:
                cat_vars = cat_vars + ce.new_columns
                num_vars = num_vars + convert2cat_vars
            else:
                cat_vars = cat_vars + convert2cat_vars

        logger.debug(f'{len(cat_vars)} categorical variables and {len(num_vars)} continuous variables found. '
                     f'{len(convert2cat_vars)} of them are from continuous to categorical.')

        self.__append_categorical_cols([(c[0], c[2] + 2) for c in cat_vars])
        self.__append_continuous_cols([c[0] for c in num_vars], consts.INPUT_PREFIX_NUM + 'all')
        print(f'Preparing features cost:{time.time() - start}')
        return X

    def _imputation(self, X):
        start = time.time()
        logger.info('Data imputation...')
        continuous_vars = self.get_continuous_columns()
        categorical_vars = self.get_categorical_columns()
        ct = ColumnTransformer([
            ('categorical', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='nan-fillvalue'),
             categorical_vars),
            ('continuous', SimpleImputer(missing_values=np.nan, strategy='mean'), continuous_vars),
        ])
        dfwrapper = DataFrameWrapper(ct, categorical_vars + continuous_vars)
        X = dfwrapper.fit_transform(X)
        self.X_transformers['imputation'] = dfwrapper
        print(f'Imputation cost:{time.time() - start}')
        return X

    def _categorical_encoding(self, X):
        start = time.time()
        logger.info('Categorical encoding...')
        vars = self.get_categorical_columns()
        mle = MultiLabelEncoder(vars)
        X = mle.fit_transform(X)
        self.X_transformers['label_encoder'] = mle
        print(f'Categorical encoding cost:{time.time() - start}')
        return X

    def _discretization(self, X):
        start = time.time()
        logger.info('Data discretization...')
        vars = self.get_continuous_columns()
        mkbd = MultiKBinsDiscretizer(vars)
        X = mkbd.fit_transform(X)
        self.__append_categorical_cols([(new_name, bins + 1) for name, new_name, bins in mkbd.new_columns])
        self.X_transformers['discreter'] = mkbd
        print(f'Discretization cost:{time.time() - start}')
        return X

    def _apply_gbm_features(self, X, y):
        start = time.time()
        logger.info('Extracting GBM features...')
        cont_vars = self.get_continuous_columns()
        cat_vars = self.get_categorical_columns()
        gbmencoder = LgbmLeavesEncoder(cat_vars, cont_vars, self.task_, **self.config.gbm_params)
        X = gbmencoder.fit_transform(X, y)
        self.X_transformers['gbm_features'] = gbmencoder
        if self.config.gbm_feature_type == consts.GBM_FEATURE_TYPE_EMB:
            self.__append_categorical_cols([(name, X[name].max() + 1) for name in gbmencoder.new_columns])
        else:
            self.__append_continuous_cols([name for name in gbmencoder.new_columns],
                                          consts.INPUT_PREFIX_NUM + 'gbm_leaves')
        print(f'Extracting gbm features cost:{time.time() - start}')
        return X

    def __append_categorical_cols(self, cols):
        logger.debug(f'{len(cols)} categorical variables appended.')

        if self.config.fixed_embedding_dim:
            embedding_output_dim = self.config.embeddings_output_dim if self.config.embeddings_output_dim > 0 else consts.EMBEDDING_OUT_DIM_DEFAULT
        else:
            embedding_output_dim = 0
            #

        if self.categorical_columns is None:
            self.categorical_columns = []
        self.categorical_columns = self.categorical_columns + \
                                   [CategoricalColumn(name,
                                                      voc_size,
                                                      embedding_output_dim
                                                      if embedding_output_dim > 0
                                                      else min(4 * int(pow(voc_size, 0.25)), 20))
                                    for name, voc_size in cols]

    def __append_continuous_cols(self, cols, input_name):
        if self.continuous_columns is None:
            self.continuous_columns = []
        self.continuous_columns = self.continuous_columns + [ContinuousColumn(name=input_name,
                                                                              column_names=[c for c in cols])]

    def get_categorical_columns(self):
        return [c.name for c in self.categorical_columns]

    def get_continuous_columns(self):
        cont_vars = []
        for c in self.continuous_columns:
            cont_vars = cont_vars + c.column_names
        return cont_vars
