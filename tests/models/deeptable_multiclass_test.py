# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from deeptables.models import deeptable
from deeptables.datasets import dsutils


class Test_DeepTable_Multiclass:
    def setup_class(self):
        print("Loading datasets...")
        data = dsutils.load_glass_uci()
        self.y = data.pop(10).values
        self.X = data

        conf = deeptable.ModelConfig(metrics=['AUC'], apply_gbm_features=False, )
        self.dt = deeptable.DeepTable(config=conf)
        self.X_train, \
        self.X_test, \
        self.y_train, \
        self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model, self.history = self.dt.fit(self.X_train, self.y_train, epochs=1)

    def teardown_class(self):
        print("Class teardown.")

    def test_class_weights(self):
        conf = deeptable.ModelConfig(metrics=['AUC'], apply_gbm_features=False, apply_class_weight=True)
        dt = deeptable.DeepTable(config=conf)
        model, history = dt.fit(self.X_train, self.y_train, epochs=1)
        assert history.history['AUC'][0] > 0

    def test_evaluate(self):
        result = self.dt.evaluate(self.X_test, self.y_test)
        assert result['AUC'] > 0

    def test_predict(self):
        preds = self.dt.predict(self.X_test)
        assert len(preds.shape) == 1

    def test_predict_proba(self):
        proba = self.dt.predict_proba(self.X_test)
        auc = roc_auc_score(self.y_test, proba, multi_class='ovo')  # ovr
        assert proba.shape[1] == 6
        assert auc > 0

    def test_proba2predict(self):
        proba = self.dt.predict_proba(self.X_test)
        preds = self.dt.predict(self.X_test)
        preds2 = self.dt.proba2predict(proba)
        auc = roc_auc_score(self.y_test, proba, multi_class='ovo')  # ovr

        assert proba.shape[1] == 6
        assert (preds == preds2).sum(), 43
        assert preds2.shape, (43,)
        assert auc > 0
