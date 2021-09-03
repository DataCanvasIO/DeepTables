# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import dask.dataframe as dd

from hypernets.tests.tabular.dask_transofromer_test import setup_dask
from .deeptable_cv_test import Test_DeepTable_CV


class Test_DeepTable_CV_Dask(Test_DeepTable_CV):
    @staticmethod
    def load_data():
        setup_dask(None)
        df = Test_DeepTable_CV.load_data()
        return dd.from_pandas(df, npartitions=2)
