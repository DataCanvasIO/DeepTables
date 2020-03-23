# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from sklearn.model_selection import train_test_split
from deeptables.utils import consts
from deeptables.models import deeptable
from deeptables.datasets import dsutils
import tempfile
import os

class Test_DeepTable_CV:
    def setup_class(self):
        print("Loading datasets...")
        df_train = dsutils.load_adult().head(1000)
        self.y = df_train.pop(14).values
        self.X = df_train

        conf = deeptable.ModelConfig(metrics=['AUC'],
                                     apply_gbm_features=False,
                                     auto_categorization=False,
                                     auto_discrete=False)
        self.dt = deeptable.DeepTable(config=conf)

        self.X_train, \
        self.X_eval, \
        self.y_train, \
        self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.oof_proba, self.eval_proba, self.test_proba = self.dt.fit_cross_validation(self.X_train,
                                                                                        self.y_train,
                                                                                        self.X_eval,
                                                                                        num_folds=3,
                                                                                        epochs=1,
                                                                                        n_jobs=1)

    def teardown_class(self):
        print("Class teardown.")

    def test_evaluate(self):
        result = self.dt.evaluate(self.X_eval, self.y_test)
        assert result['AUC'] > 0

    def test_best_model(self):
        model = self.dt.best_model
        # print(model.summary())
        assert model

    def test_oof_proba(self):
        oof_predict = self.dt.proba2predict(self.oof_proba)
        assert oof_predict.shape, (1000,)

    def test_test_proba(self):
        test_predict = self.dt.proba2predict(self.eval_proba)
        assert test_predict.shape, (200,)

    def test_predict_proba_all_model_avg(self):
        proba = self.dt.predict_proba(self.X_eval, model_selector=consts.MODEL_SELECTOR_ALL)
        assert proba.shape, (200, 1)

    def test_predict_proba_all_model(self):
        proba_all = self.dt.predict_proba_all(self.X_eval)
        assert len(proba_all), 3
        assert proba_all['dnn_nets-kfold-1'].shape, (200, 1)

    def test_predict_proba(self):
        proba = self.dt.predict_proba(self.X_eval)
        assert proba.shape, (200, 1)

    def test_proba2predict(self):
        proba = self.dt.predict_proba(self.X_eval)
        preds = self.dt.predict(self.X_eval)
        preds2 = self.dt.proba2predict(proba)
        assert proba.shape, (200, 1)
        assert all(preds == preds2)
        assert preds2.shape, (200,)

    def test_get_model(self):
        best = self.dt.get_model(model_selector=consts.MODEL_SELECTOR_BEST)
        current = self.dt.get_model(model_selector=consts.MODEL_SELECTOR_CURRENT)
        byname = self.dt.get_model(model_selector='dnn_nets-kfold-1')
        assert best
        assert current
        assert byname

    def test_save_load(self):
        filepath = tempfile.mkdtemp()
        self.dt.save(filepath)
        assert os.path.exists(f'{filepath}/dt.pkl')
        assert os.path.exists(f'{filepath}/dnn_nets-kfold-1.h5')
        assert os.path.exists(f'{filepath}/dnn_nets-kfold-2.h5')
        assert os.path.exists(f'{filepath}/dnn_nets-kfold-3.h5')
        newdt = deeptable.DeepTable.load(filepath)
        preds = newdt.predict(self.X_eval)
        assert preds.shape, (200,)

if __name__ == "__main__":
    pass
