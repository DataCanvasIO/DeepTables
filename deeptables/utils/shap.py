# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import shap
import pandas as pd


class DeepTablesExplainer():
    def __init__(self, dt_model, X, num_samples=50):
        self.dt_model = dt_model
        self.num_samples = num_samples
        self.feature_names = X.columns.to_list()

        if num_samples is not None:
            samples = shap.sample(X, num_samples)
        else:
            samples = X
        self.explainer = shap.KernelExplainer(self.model_fn, samples)

    def model_fn(self, data_asarray):
        data_asframe = pd.DataFrame(data_asarray, columns=self.feature_names)
        return self.dt_model.predict(data_asframe, encode_to_label=False)

    def get_shap_values(self, X_to_explain, nsamples='auto'):
        shap_values = self.explainer.shap_values(X_to_explain, nsamples=nsamples)
        return shap_values

