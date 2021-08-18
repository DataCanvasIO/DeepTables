# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

try:
    from deeptables.utils.feature_importance import get_score_importances

    have_eli5 = True
except ImportError:
    have_eli5 = False

from deeptables.models import deeptable
from deeptables.datasets import dsutils
from sklearn.model_selection import train_test_split


class Test_Importances():
    def test_importances(self):
        if have_eli5:
            df = dsutils.load_bank().head(100)
            df.drop(['id'], axis=1, inplace=True)
            X, X_test = train_test_split(df, test_size=0.2, random_state=42)
            y = X.pop('y')
            y_test = X_test.pop('y')

            config = deeptable.ModelConfig(nets=['dnn_nets'], auto_discrete=True, metrics=['AUC'])
            dt = deeptable.DeepTable(config=config)
            dt.fit(X, y, epochs=1)

            fi = get_score_importances(dt, X_test, y_test, 'AUC', 1, mode='max')
            assert fi.shape == (16, 2)

            fi2 = get_score_importances(dt, X_test, y_test, 'log_loss', 1, mode='min')
            assert fi2.shape == (16, 2)
