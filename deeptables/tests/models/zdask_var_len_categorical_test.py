# -*- encoding: utf-8 -*-
from hypernets.tests.tabular.tb_dask import is_dask_installed, if_dask_ready, setup_dask
from .var_len_categorical_test import TestVarLenCategoricalFeature

if is_dask_installed:
    import dask.dataframe as dd


@if_dask_ready
class TestVarLenCategoricalFeatureByDask(TestVarLenCategoricalFeature):

    def setup_class(self):
        TestVarLenCategoricalFeature.setup_class(self)

        setup_dask(self)
        self.df = dd.from_pandas(self.df, npartitions=2)
