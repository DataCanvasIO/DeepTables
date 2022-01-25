# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from hypernets.tests.tabular.tb_dask import is_dask_installed, if_dask_ready, setup_dask
from .deeptable_cv_test import Test_DeepTable_CV

if is_dask_installed:
    import dask.dataframe as dd


@if_dask_ready
class TestDeepTableCV_Dask(Test_DeepTable_CV):
    @staticmethod
    def load_data():
        setup_dask(None)
        df = Test_DeepTable_CV.load_data()
        return dd.from_pandas(df, npartitions=2)
