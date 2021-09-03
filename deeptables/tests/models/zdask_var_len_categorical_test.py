# -*- encoding: utf-8 -*-
import dask.dataframe as dd

from hypernets.tests.tabular.dask_transofromer_test import setup_dask
from .var_len_categorical_test import TestVarLenCategoricalFeature


class TestVarLenCategoricalFeatureByDask(TestVarLenCategoricalFeature):

    def setup_class(self):
        TestVarLenCategoricalFeature.setup_class(self)

        setup_dask(self)
        self.df = dd.from_pandas(self.df, npartitions=2)
