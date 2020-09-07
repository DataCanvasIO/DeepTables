# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import numpy as np
import pandas as pd
from deeptables.models import deeptable, deepnets
from deeptables.datasets import dsutils
from sklearn.model_selection import train_test_split
import shap


class Test_SHAP():
    def test_shap(self):
        df = dsutils.load_bank().head(100)
        df.drop(['id'], axis=1, inplace=True)
        X, X_test = train_test_split(df, test_size=0.2, random_state=42)
        y = X.pop('y')
        y_test = X_test.pop('y')

        config = deeptable.ModelConfig(nets=['dnn_nets'], auto_discrete=True, metrics=['AUC'])
        dt = deeptable.DeepTable(config=config)
        dt.fit(X, y, epochs=1)
        feature_names = X.columns.to_list()

        def model_predict(data_asarray):
            data_asframe = pd.DataFrame(data_asarray, columns=feature_names)
            return dt.predict(data_asframe, encode_to_label=False)

        explainer = shap.KernelExplainer(model_predict, shap.sample(X,50))
        shap_values = explainer.shap_values(X[:1], nsamples='auto')
        shap.summary_plot(shap_values, X[:1000])
        print(shap_values)
