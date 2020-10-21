# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from sklearn.model_selection import train_test_split

from deeptables.models import deeptable
from deeptables.models.preprocessor import DefaultPreprocessor
from deeptables.datasets import dsutils
from .. import homedir


class Test_Preprocessor:
    def test_transform(self):
        df_train = dsutils.load_adult()
        y = df_train.pop(14).values
        X = df_train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        conf = deeptable.ModelConfig(auto_discrete=True,
                                     auto_imputation=True,
                                     auto_encode_label=True,
                                     auto_categorize=True,
                                     apply_gbm_features=False)
        processor = DefaultPreprocessor(conf)
        X1, y1 = processor.fit_transform(X_train, y_train)
        X2, y2 = processor.transform(X_test, y_test)
        assert len(set(X1.columns.tolist()) - set(['x_1', 'x_3', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_13', 'x_0_cat',
                                                   'x_4_cat', 'x_10_cat', 'x_11_cat', 'x_12_cat', 'x_2', 'x_0', 'x_4',
                                                   'x_10', 'x_11', 'x_12', 'x_2_discrete', 'x_0_discrete',
                                                   'x_4_discrete',
                                                   'x_10_discrete', 'x_11_discrete', 'x_12_discrete'])) == 0
        assert len(set(X1.columns) - set(X2.columns)) == 0
        assert X1.shape, (X_train.shape[0], 25)
        assert X2.shape, (X_test.shape[0], 25)
        assert y1.sum(), 6270
        assert y2.sum(), 1571

    def test_categorical_columns_config(self):
        df_train = dsutils.load_adult().head(1000)
        y = df_train.pop(14).values

        conf = deeptable.ModelConfig(
            categorical_columns=['x_1', 'x_2', 'x_3'],
            auto_discrete=False,
            auto_imputation=True,
            auto_encode_label=True,
            auto_categorize=False,
            apply_gbm_features=False)
        processor = DefaultPreprocessor(conf)
        X, y = processor.fit_transform(df_train, y)
        assert len(set(X.columns) -
                   set(['x_1', 'x_2', 'x_3', 'x_0', 'x_4', 'x_10', 'x_11', 'x_12'])) == 0

    def test_categorical_columns_config_2(self):
        df_train = dsutils.load_adult().head(1000)
        y = df_train.pop(14).values

        conf = deeptable.ModelConfig(
            categorical_columns=['x_1', 'x_2', 'x_3'],
            auto_discrete=True,
            auto_imputation=True,
            auto_encode_label=True,
            auto_categorize=False,
            apply_gbm_features=False)
        processor = DefaultPreprocessor(conf)
        X, y = processor.fit_transform(df_train, y)
        assert len(set(X.columns) -
                   set(['x_1', 'x_2', 'x_3', 'x_0', 'x_4', 'x_10', 'x_11', 'x_12',
                        'x_0_discrete', 'x_4_discrete', 'x_10_discrete', 'x_11_discrete',
                        'x_12_discrete'])) == 0

    def test_use_cache(self):
        config = deeptable.ModelConfig(metrics=['AUC'], apply_gbm_features=False, apply_class_weight=True)
        df_train = dsutils.load_adult().head(1000)
        y = df_train.pop(14).values
        X = df_train

        X, X_val, y, y_val = train_test_split(X, y, test_size=0.2)
        cache_home = homedir + '/preprocessor_cache'
        preprocessor = DefaultPreprocessor(config, cache_home=cache_home, use_cache=True)
        preprocessor.clear_cache()

        sign = preprocessor.get_X_y_signature(X, y)
        sign_val = preprocessor.get_X_y_signature(X_val, y_val)
        X_t, y_t = preprocessor.get_transformed_X_y_from_cache(sign)

        assert X_t is None and y_t is None

        preprocessor.fit_transform(X, y)
        preprocessor.transform(X_val, y_val)
        X_t2, y_t2 = preprocessor.get_transformed_X_y_from_cache(sign)
        assert X_t2 is not None and y_t2 is not None

        preprocessor = DefaultPreprocessor(config, cache_home=cache_home, use_cache=True)

        assert len(preprocessor.X_transformers) == 0
        assert preprocessor.y_lable_encoder is None

        assert preprocessor.load_transformers_from_cache() == True

        assert len(preprocessor.X_transformers) == 3
        assert preprocessor.y_lable_encoder is not None

        X_t, y_t = preprocessor.get_transformed_X_y_from_cache(sign)
        assert X_t is not None and y_t is not None

        X_val_t, y_val_t = preprocessor.get_transformed_X_y_from_cache(sign_val)
        assert X_val_t is not None and y_val_t is not None
