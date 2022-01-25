# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import pandas as pd

from hypernets.tests.tabular.tb_dask import is_dask_installed, if_dask_ready, setup_dask
from .deeptable_regression_test import Test_DeepTable_Regression

if is_dask_installed:
    import dask.dataframe as dd


@if_dask_ready
class TestDeepTableRegressionByDask(Test_DeepTable_Regression):
    @staticmethod
    def load_data():
        X, y = Test_DeepTable_Regression.load_data()
        df = pd.concat([X, y], axis=1, ignore_index=True)
        df.columns = X.columns.to_list() + ['target']
        ddf = dd.from_pandas(df, npartitions=2)
        yt = ddf.pop('target')
        return ddf, yt

    def setup_class(self):
        setup_dask(self)
        super().setup_class(self)
