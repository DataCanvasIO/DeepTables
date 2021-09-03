# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import pandas as pd
from sklearn.datasets import load_boston

from deeptables.models import deeptable
from deeptables.tests.misc import r2_c
from deeptables.utils import consts
from hypernets.tabular.dask_ex import train_test_split


class Test_DeepTable_Regression:

    @staticmethod
    def load_data():
        print("Loading datasets...")

        boston_dataset = load_boston()

        df_train = pd.DataFrame(boston_dataset.data)
        df_train.columns = boston_dataset.feature_names
        target = pd.Series(boston_dataset.target)

        return df_train, target

    def setup_class(self):
        self.X, self.y = self.load_data()

        conf = deeptable.ModelConfig(task=consts.TASK_REGRESSION, metrics=[r2_c, 'RootMeanSquaredError'],
                                     apply_gbm_features=False)
        self.dt = deeptable.DeepTable(config=conf)

        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model, self.history = self.dt.fit(self.X_train, self.y_train, batch_size=32, epochs=100)

    def test_leaderboard(self):
        lb = self.dt.leaderboard
        assert lb.shape, (1, 8)

    def teardown_class(self):
        print("Class teardown.")

    def test_evaluate(self):
        result = self.dt.evaluate(self.X_test, self.y_test)
        score = result.get('RootMeanSquaredError')
        if score is None:
            score = result.get('Root_Mean_Squared_Error')  # for tf v2.2+

        assert score
        assert score > 0

    def test_task(self):
        assert self.dt.task == consts.TASK_REGRESSION
