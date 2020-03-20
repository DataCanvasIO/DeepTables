# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from deeptables.models import modelset
from deeptables.models import deeptable
from deeptables.datasets import dsutils
import deeptables.models as deepmodels


class Test_ModelSet:
    def test_best_model(self):
        ms = modelset.ModelSet()
        auc_best = modelset.ModelInfo('val','m1', 'aucbest', {'AUC': 0.9, 'acc': 0.8, 'mse': 0.2})
        acc_best = modelset.ModelInfo('val','m2', 'accbest', {'AUC': 0.8, 'acc': 0.9, 'mse': 0.3})
        mse_best = modelset.ModelInfo('val','m3', 'msebest', {'AUC': 0.8, 'acc': 0.83, 'mse': 0.1})

        ms.push(auc_best)
        ms.push(acc_best)
        ms.push(mse_best)

        ms.metric = 'AUC'
        ms.best_mode = 'max'
        bm = ms.best_model()
        assert bm.model == 'aucbest'

        ms.metric = 'acc'
        ms.best_mode = 'max'
        bm = ms.best_model()
        assert bm.model == 'accbest'

        ms.metric = 'mse'
        ms.best_mode = 'min'
        bm = ms.best_model()
        assert bm.model == 'msebest'

        # auto mode
        ms.metric = 'AUC'
        ms.best_mode = 'auto'
        bm = ms.best_model()
        assert bm.model == 'aucbest'

        ms.metric = 'acc'
        ms.best_mode = 'auto'
        bm = ms.best_model()
        assert bm.model == 'accbest'

        ms.metric = 'mse'
        ms.best_mode = 'auto'
        bm = ms.best_model()
        assert bm.model == 'msebest'

    def test_model_meta(self):
        m = modelset.ModelInfo('val','m1', 'model', {'AUC': 0.9}, modelname='m1', modelversion=1.0)
        assert m.meta['modelname'] == 'm1'
        assert m.meta['modelversion'] == 1.0

        meta = {'modelname': 'm1', 'modelversion': 1.0}
        m = modelset.ModelInfo('val','m1', 'model', {'AUC': 0.9}, **meta)
        assert m.meta['modelname'] == 'm1'
        assert m.meta['modelversion'] == 1.0

    def test_modelinfo(self):
        df_train = dsutils.load_adult()
        y = df_train.pop(14).values
        X = df_train

        conf = deepmodels.ModelConfig(metrics=['AUC'])
        dt = deeptable.DeepTable(config=conf)
        model, history = dt.fit(X, y, epochs=2)
        mi = modelset.ModelInfo('val','m1', model, {}, history=history.history)
        assert mi.score['val_auc'] > 0
        assert len(mi.meta['history']['AUC']) == 2


if __name__ == "__main__":
    pass


