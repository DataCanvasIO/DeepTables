# -*- coding:utf-8 -*-
"""

"""
from deeptables.datasets import dsutils
from deeptables.models import deeptable
from hypernets.tabular import get_tool_box
from hypernets.tests.tabular.tb_dask import is_dask_installed, if_dask_ready, setup_dask
from .deeptable_multiclass_test import Test_DeepTable_Multiclass

if is_dask_installed:
    import dask.dataframe as dd


@if_dask_ready
class TestDeepTableMulticlassByDask(Test_DeepTable_Multiclass):
    def setup_class(self):
        setup_dask(self)

        print("Loading datasets...")
        data = dd.from_pandas(dsutils.load_glass_uci(), npartitions=2)
        self.y = data.pop(10).values
        self.X = data

        conf = deeptable.ModelConfig(metrics=['AUC'], apply_gbm_features=False, )
        self.dt = deeptable.DeepTable(config=conf)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            [t.persist() for t in get_tool_box(data).train_test_split(self.X, self.y, test_size=0.2, random_state=42)]
        self.model, self.history = self.dt.fit(self.X_train, self.y_train, batch_size=32, epochs=3)
