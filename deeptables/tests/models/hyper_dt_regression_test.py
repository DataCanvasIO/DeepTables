# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import pandas as pd
from deeptables.models import DeepTable
from deeptables.models.hyper_dt import HyperDT, tiny_dt_space
from hypernets.core.callbacks import SummaryCallback, FileStorageLoggingCallback
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers import RandomSearcher

from sklearn.model_selection import train_test_split
from hypernets.tabular.datasets.dsutils import load_boston

from .. import homedir


class Test_HyperDT_Regression():

    def test_boston(self):
        print("Loading datasets...")
        df = load_boston()
        df_train = df
        self.y = df.pop('target')
        self.X = df_train

        self.X_train, \
        self.X_test, \
        self.y_train, \
        self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        rs = RandomSearcher(tiny_dt_space, optimize_direction=OptimizeDirection.Maximize, )
        hdt = HyperDT(rs,
                      callbacks=[SummaryCallback(), FileStorageLoggingCallback(rs, output_dir=f'{homedir}/hyn_logs')],
                      reward_metric='RootMeanSquaredError',
                      dnn_params={
                          'hidden_units': ((256, 0, False), (256, 0, False)),
                          'dnn_activation': 'relu',
                      },
                      )
        hdt.search(self.X_train, self.y_train, self.X_test, self.y_test, max_trials=3)

        best_trial = hdt.get_best_trial()

        estimator = hdt.final_train(best_trial.space_sample, self.X, self.y)
        score = estimator.predict(self.X_test)
        result = estimator.evaluate(self.X_test, self.y_test)
        assert result
        assert isinstance(estimator.model, DeepTable)
