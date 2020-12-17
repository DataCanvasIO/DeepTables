# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

try:
    import shap
    from deeptables.utils.shap import DeepTablesExplainer
    have_shap = True
except ImportError:
    have_shap = False

from deeptables.models import deeptable
from deeptables.datasets import dsutils
from sklearn.model_selection import train_test_split


class Test_SHAP():
    def test_shap(self):
        if have_shap:
            df = dsutils.load_bank().head(100)
            df.drop(['id'], axis=1, inplace=True)
            X, X_test = train_test_split(df, test_size=0.2, random_state=42)
            y = X.pop('y')
            y_test = X_test.pop('y')

            config = deeptable.ModelConfig(nets=['dnn_nets'], auto_discrete=True, metrics=['AUC'])
            dt = deeptable.DeepTable(config=config)
            dt.fit(X, y, epochs=1)

            dt_explainer = DeepTablesExplainer(dt, X, num_samples=10)

            shap_values = dt_explainer.get_shap_values(X[:1], nsamples='auto')
            assert shap_values[0].shape == (16, )
