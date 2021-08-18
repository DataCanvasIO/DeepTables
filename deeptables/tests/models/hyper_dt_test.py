# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import numpy as np
import pandas as pd
from deeptables.datasets import dsutils
from deeptables.models import DeepTable
from deeptables.models.hyper_dt import HyperDT, DTEstimator, mini_dt_space, tiny_dt_space, default_dt_space
from hypernets.core.callbacks import SummaryCallback, FileStorageLoggingCallback
from hypernets.core.searcher import OptimizeDirection
from hypernets.searchers import RandomSearcher
from sklearn.model_selection import train_test_split

from .. import homedir


class Test_HyperDT():
    def test_bankdata(self):
        rs = RandomSearcher(tiny_dt_space, optimize_direction=OptimizeDirection.Maximize, )
        hdt = HyperDT(rs,
                      callbacks=[SummaryCallback(), FileStorageLoggingCallback(rs, output_dir=f'{homedir}/hyn_logs')],
                      # reward_metric='accuracy',
                      reward_metric='AUC',
                      dnn_params={
                          'hidden_units': ((256, 0, False), (256, 0, False)),
                          'dnn_activation': 'relu',
                      },
                      )

        df = dsutils.load_bank().sample(n=2000, random_state=9527)
        df.drop(['id'], axis=1, inplace=True)
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
        y = df_train.pop('y')
        y_test = df_test.pop('y')

        hdt.search(df_train, y, df_test, y_test, max_trials=3, )
        best_trial = hdt.get_best_trial()
        assert best_trial

        estimator = hdt.final_train(best_trial.space_sample, df_train, y)
        score = estimator.predict(df_test)
        result = estimator.evaluate(df_test, y_test)
        assert len(score) == len(y_test)
        assert result
        assert isinstance(estimator.model, DeepTable)

    def test_default_dt_space(self):
        space = default_dt_space()
        space.random_sample()
        assert space.Module_DnnModule_1.param_values['dnn_layers'] == len(
            space.DT_Module.config.dnn_params['hidden_units'])
        assert space.Module_DnnModule_1.param_values['hidden_units'] == \
               space.DT_Module.config.dnn_params['hidden_units'][0][
                   0]
        assert space.Module_DnnModule_1.param_values['dnn_dropout'] == \
               space.DT_Module.config.dnn_params['hidden_units'][0][
                   1]
        assert space.Module_DnnModule_1.param_values['use_bn'] == space.DT_Module.config.dnn_params['hidden_units'][0][
            2]

    def test_hyper_dt(self):
        rs = RandomSearcher(tiny_dt_space, optimize_direction=OptimizeDirection.Maximize, )
        hdt = HyperDT(rs,
                      callbacks=[SummaryCallback()],
                      reward_metric='accuracy',
                      dnn_params={
                          'hidden_units': ((256, 0, False), (256, 0, False)),
                          'dnn_activation': 'relu',
                      },
                      cache_preprocessed_data=True,
                      )
        x1 = np.random.randint(0, 10, size=(100), dtype='int')
        x2 = np.random.randint(0, 2, size=(100)).astype('str')
        x3 = np.random.randint(0, 2, size=(100)).astype('str')
        x4 = np.random.normal(0.0, 1.0, size=(100))

        y = np.random.randint(0, 2, size=(100), dtype='int')
        df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})
        hdt.search(df, y, df, y, max_trials=3, epochs=1)
        best_trial = hdt.get_best_trial()
        model = hdt.load_estimator(best_trial.model_file)
        assert model
        score = model.predict(df)
        result = model.evaluate(df, y)
        assert len(score) == 100
        assert result
        assert isinstance(model, DTEstimator)

        estimator = hdt.final_train(best_trial.space_sample, df, y, epochs=1)
        score = estimator.predict(df)
        result = estimator.evaluate(df, y)
        assert len(score) == 100
        assert result
        assert isinstance(estimator, DTEstimator)
        assert isinstance(estimator.model, DeepTable)
