# -*- coding:utf-8 -*-
"""

"""
from deeptables.datasets import dsutils
from deeptables.models import deeptable
from hypernets.tabular import get_tool_box
from hypernets.tests.tabular.tb_dask import is_dask_installed, if_dask_ready, setup_dask

if is_dask_installed:
    import dask.dataframe as dd
    from deeptables.models.preprocessor import DefaultDaskPreprocessor


@if_dask_ready
class Test_Preprocessor_Dask:
    @classmethod
    def setup_class(cls):
        setup_dask(cls)

    def test_transform(self):
        df_train = dsutils.load_adult()
        df_train = dd.from_pandas(df_train, npartitions=2)
        y = df_train.pop(14)  # .values
        X = df_train
        X_train, X_test, y_train, y_test = get_tool_box(X, y).train_test_split(X, y, test_size=0.2, random_state=42)
        conf = deeptable.ModelConfig(auto_discrete=True,
                                     auto_imputation=True,
                                     auto_encode_label=True,
                                     auto_categorize=True,
                                     apply_gbm_features=False)
        processor = DefaultDaskPreprocessor(conf, compute_to_local=True)
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
        assert y1.sum(), 6297
        assert y2.sum(), 1544

    def test_categorical_columns_config(self):
        df_train = dsutils.load_adult().head(1000)
        df_train = dd.from_pandas(df_train, npartitions=2)
        y = df_train.pop(14).values

        conf = deeptable.ModelConfig(
            categorical_columns=['x_1', 'x_2', 'x_3'],
            auto_discrete=False,
            auto_imputation=True,
            auto_encode_label=True,
            auto_categorize=False,
            apply_gbm_features=False)
        processor = DefaultDaskPreprocessor(conf, compute_to_local=True)
        X, y = processor.fit_transform(df_train, y)
        assert len(set(X.columns) -
                   set(['x_1', 'x_2', 'x_3', 'x_0', 'x_4', 'x_10', 'x_11', 'x_12'])) == 0

    def test_categorical_columns_config_2(self):
        df_train = dsutils.load_adult().head(1000)
        df_train = dd.from_pandas(df_train, npartitions=2)
        y = df_train.pop(14)

        conf = deeptable.ModelConfig(
            categorical_columns=['x_1', 'x_2', 'x_3'],
            auto_discrete=True,
            auto_imputation=True,
            auto_encode_label=True,
            auto_categorize=False,
            apply_gbm_features=False)
        processor = DefaultDaskPreprocessor(conf, compute_to_local=True)
        X, y = processor.fit_transform(df_train, y)
        assert len(set(X.columns) -
                   set(['x_1', 'x_2', 'x_3', 'x_0', 'x_4', 'x_10', 'x_11', 'x_12',
                        'x_0_discrete', 'x_4_discrete', 'x_10_discrete', 'x_11_discrete',
                        'x_12_discrete'])) == 0


if __name__ == '__main__':
    pass
