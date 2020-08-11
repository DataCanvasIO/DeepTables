# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from deeptables.models import deeptable
from tests.misc import r2_c
from deeptables.utils import consts


class Test_DeepTable_Regression:
    def setup_class(self):
        print("Loading datasets...")
        boston_dataset = load_boston()

        df_train = pd.DataFrame(boston_dataset.data)
        df_train.columns = boston_dataset.feature_names
        self.y = pd.Series(boston_dataset.target)
        self.X = df_train

        conf = deeptable.ModelConfig(task=consts.TASK_REGRESSION, metrics=[r2_c, 'RootMeanSquaredError'], apply_gbm_features=False)
        self.dt = deeptable.DeepTable(config=conf)

        self.X_train, \
        self.X_test, \
        self.y_train, \
        self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model, self.history = self.dt.fit(self.X_train, self.y_train, epochs=100)

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
