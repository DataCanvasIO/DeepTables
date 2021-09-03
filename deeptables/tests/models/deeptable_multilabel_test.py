# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from deeptables.models import deeptable
from deeptables.datasets import dsutils


class Test_DeepTable_MultiLabel:
    def test_fit(self):
        print("Loading datasets...")
        x1 = np.random.randint(0, 10, size=(100), dtype='int')
        x2 = np.random.normal(0.0, 1.0, size=(100))
        x3 = np.random.normal(0.0, 1.0, size=(100))

        y1 = np.random.randint(0, 2, size=(100), dtype='int')
        y2 = np.random.randint(0, 2, size=(100), dtype='int')

        df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
        df_y = pd.DataFrame({'y1': y1, 'y2': y2})

        conf = deeptable.ModelConfig(metrics=['AUC'], nets=['dnn_nets'], apply_gbm_features=False, task='multilabel')
        dt = deeptable.DeepTable(config=conf)
        X_train, X_test, y_train, y_test = train_test_split(df, df_y, test_size=0.2, random_state=42)
        model, history = dt.fit(X_train, y_train.values, batch_size=10, epochs=1)

    def test_fit_cross_validation(self):
        print("Loading datasets...")
        x1 = np.random.randint(0, 10, size=(100), dtype='int')
        x2 = np.random.normal(0.0, 1.0, size=(100))
        x3 = np.random.normal(0.0, 1.0, size=(100))

        y1 = np.random.randint(0, 2, size=(100), dtype='int')
        y2 = np.random.randint(0, 2, size=(100), dtype='int')

        df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
        df_y = pd.DataFrame({'y1': y1, 'y2': y2})

        conf = deeptable.ModelConfig(metrics=['AUC'], nets=['dnn_nets'], apply_gbm_features=False, task='multilabel')
        dt = deeptable.DeepTable(config=conf)
        oof_predict, _, test_predict = dt.fit_cross_validation(df, df_y, X_test=df, num_folds=3)
        assert oof_predict.shape[-1] == df_y.shape[-1]
        assert test_predict.shape[-1] == df_y.shape[-1]
