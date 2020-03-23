# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import random
import numpy as np
import pandas as pd
import os
import tempfile

from sklearn import manifold
from sklearn.metrics import roc_auc_score, mean_squared_error, f1_score
from sklearn.model_selection import train_test_split

from deeptables.models import deeptable, deepmodel
from deeptables.utils import consts
from deeptables.datasets import dsutils
import pytest


class Test_DeepTable:
    def setup_class(self):
        print("Loading datasets...")
        df_train = dsutils.load_adult().head(1000)
        self.y = df_train.pop(14).values
        self.X = df_train

        conf = deeptable.ModelConfig(metrics=['AUC'], apply_gbm_features=False, apply_class_weight=True)
        self.dt = deeptable.DeepTable(config=conf)

        self.X_train, \
        self.X_test, \
        self.y_train, \
        self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model, self.history = self.dt.fit(self.X_train, self.y_train, epochs=1)

    def teardown_class(self):
        print("Class teardown.")

    def test_evaluate(self):
        result = self.dt.evaluate(self.X_test, self.y_test)
        assert result['AUC'] > 0

    def test_model_selector(self):
        m1 = self.dt.get_model(consts.MODEL_SELECTOR_CURRENT)
        m2 = self.dt.get_model(consts.MODEL_SELECTOR_BEST)
        m3 = self.dt.get_model('dnn_nets')

        assert isinstance(m1, deepmodel.DeepModel)
        assert m1 is m2
        assert m2 is m3

    def test_best_model(self):
        model = self.dt.best_model
        assert isinstance(model, deepmodel.DeepModel)

    def test_predict_proba(self):
        proba = self.dt.predict_proba(self.X_test)
        assert proba.shape, (6513, 1)

    def test_proba2predict(self):
        proba = self.dt.predict_proba(self.X_test)
        preds = self.dt.predict(self.X_test)
        preds2 = self.dt.proba2predict(proba)
        assert proba.shape, (6513, 1)
        assert (preds == preds2).sum(), 6513
        assert preds2.shape, (6513,)

    def test_apply(self):
        features = self.dt.apply(self.X_test,
                                 output_layers=['flatten_embeddings', 'dnn_dense_1', 'dnn_dense_2'])
        assert len(features) == 3
        assert len(features[0].shape) == 2

        features = self.dt.apply(self.X_test, output_layers=['flatten_embeddings'])
        assert len(features.shape) == 2

    def test_apply_with_transformer(self):
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

        out1 = random.sample(range(self.X_test.shape[0]), 100)
        X_sample = self.X_test.iloc[out1,]
        y_sample = self.y_test[out1]

        features = self.dt.apply(X_sample,
                                 output_layers=['flatten_embeddings', 'dnn_dense_1'],
                                 transformer=tsne)
        assert len(features) == 2
        assert len(features[0].shape) == 2
        assert features[0].shape[1] == 2
        assert features[1].shape[1] == 2

    def test_probe_evaluate(self):
        result = deeptable.probe_evaluate(self.dt, self.X_train, self.y_train, self.X_test, self.y_test,
                                          layers=['flatten_embeddings'], score_fn={})

        assert len(result) == 1
        assert result['flatten_embeddings']['accuracy'] > 0

        result = deeptable.probe_evaluate(self.dt, self.X_train, self.y_train, self.X_test, self.y_test,
                                          layers=['flatten_embeddings', 'dnn_dense_1', 'dnn_dense_2'],
                                          score_fn={'AUC': roc_auc_score, 'F1': f1_score, 'MSE': mean_squared_error})

        assert len(result) == 3
        assert len(result['flatten_embeddings']) == 3
        assert result['flatten_embeddings']['AUC'] > 0
        assert result['dnn_dense_2']['AUC'] > 0

    # def test_gbm_params(self):
    #     df_train = dsutils.load_adult().head(1000)
    #     y = df_train.pop(14).values
    #     X = df_train
    #
    #     conf = deeptable.ModelConfig(metrics=['AUC'],
    #                                  apply_gbm_features=True,
    #                                  gbm_params={'learning_rate': 0.01, 'colsample_bytree': 0.95, 'reg_alpha': 0.04,
    #                                              'reg_lambda': 0.07},
    #                                  )
    #     dt = deeptable.DeepTable(config=conf)
    #     model, history = self.dt.fit(X, y, epochs=1)
    #     lgbm_encoder = dt.get_transformer('gbm_features')
    #     assert lgbm_encoder.lgbm.learning_rate, 0.01
    #     assert lgbm_encoder.lgbm.colsample_bytree, 0.95
    #     assert lgbm_encoder.lgbm.reg_alpha, 0.04
    #     assert lgbm_encoder.lgbm.reg_lambda, 0.07

    def test_gbm_feature_embedding(self):
        df_train = dsutils.load_adult().head(1000)
        y = df_train.pop(14).values
        X = df_train

        conf = deeptable.ModelConfig(metrics=['AUC'],
                                     apply_gbm_features=True,
                                     gbm_feature_type=consts.GBM_FEATURE_TYPE_EMB,
                                     gbm_params={'learning_rate': 0.01, 'colsample_bytree': 0.95, 'reg_alpha': 0.04,
                                                 'reg_lambda': 0.07, 'n_estimators': 10},
                                     )
        dt = deeptable.DeepTable(config=conf)
        dm, history = dt.fit(X, y, epochs=1)
        lgbm_leaves = [c for c in dt.preprocessor.get_categorical_columns() if 'lgbm_leaf' in c]
        assert len(lgbm_leaves), 10

    def test_gbm_feature_dense(self):
        df_train = dsutils.load_adult().head(1000)
        y = df_train.pop(14).values
        X = df_train

        conf = deeptable.ModelConfig(metrics=['AUC'],
                                     apply_gbm_features=True,
                                     gbm_feature_type=consts.GBM_FEATURE_TYPE_DENSE,
                                     gbm_params={'learning_rate': 0.01, 'colsample_bytree': 0.95, 'reg_alpha': 0.04,
                                                 'reg_lambda': 0.07, 'n_estimators': 10},
                                     )
        dt = deeptable.DeepTable(config=conf)
        dm, history = dt.fit(X, y, epochs=1)
        layers = dm.model.layers
        dense_lgbm_input = dm.model.get_layer(consts.INPUT_PREFIX_NUM + 'gbm_leaves')
        concat_continuous_inputs = dm.model.get_layer('concat_continuous_inputs')
        # last_lgbm_emb = model.get_layer('emb_lgbm_leaf_9')
        # flatten_embeddings = model.get_layer('flatten_embeddings')
        assert dense_lgbm_input
        assert concat_continuous_inputs

    def test_predict_unseen_data(self):
        x1 = np.random.randint(0, 10, size=(100), dtype='int')
        x2 = np.random.randint(0, 2, size=(100)).astype('str')
        x3 = np.random.normal(0.0, 1.0, size=(100))

        y = np.random.randint(0, 2, size=(100), dtype='int')
        df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
        dt = deeptable.DeepTable(config=deeptable.ModelConfig(
            apply_gbm_features=False,
            auto_categorization=True,
            auto_discrete=True,
            # nets=['linear', 'cin_nets', 'fm_nets', 'afm_nets', 'pnn_nets', 'dnn2_nets', 'dcn_nets',
            #       'autoint_nets', 'fibi_dnn_nets'],
            # 'fg_nets', 'fgcnn_cin_nets', 'fgcnn_fm_nets', 'fgcnn_ipnn_nets',
            #          'fgcnn_dnn_nets', ]
        ))
        dt.fit(df, y)
        xt_1 = np.random.randint(0, 50, size=(10), dtype='int')
        xt_2 = np.random.randint(0, 10, size=(10)).astype('str')
        xt_3 = np.random.normal(0.0, 2.0, size=(10))

        dft = pd.DataFrame({'x1': xt_1, 'x2': xt_2, 'x3': xt_3})
        preds = dt.predict(dft)
        assert len(preds), 10

    def test_infer_task_type(self):
        y1 = np.random.randint(0, 2, size=(1000), dtype='int')
        y2 = np.random.randint(0, 2, size=(1000)).astype('str')
        y3 = np.random.randint(0, 20, size=(1000)).astype('object')
        y4 = np.random.random(size=(1000)).astype('float')

        task, _ = deeptable.infer_task_type(y1)
        assert (task, consts.TASK_BINARY)

        task, _ = deeptable.infer_task_type(y2)
        assert (task, consts.TASK_BINARY)

        task, _ = deeptable.infer_task_type(y3)
        assert (task, consts.TASK_MULTICLASS)

        task, _ = deeptable.infer_task_type(y4)
        assert (task, consts.TASK_REGRESSION)

    def test_duplicate_columns(self):
        x1 = np.random.randint(0, 10, size=(100), dtype='int')
        x2 = np.random.randint(0, 2, size=(100)).astype('str')
        x3 = np.random.normal(0.0, 1.0, size=(100))

        y = np.random.randint(0, 2, size=(100), dtype='int')
        df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})
        df.columns = ['x1', 'x1', 'x3']
        dt = deeptable.DeepTable(config=deeptable.ModelConfig(
            apply_gbm_features=False,
            auto_categorization=True,
            auto_discrete=True,
        ))
        with pytest.raises(ValueError) as excinfo:
            dt.fit(df, y)
        assert "Columns with duplicate names in X:" in str(excinfo.value)
        assert excinfo.type == ValueError

    def test_save_load(self):
        filepath = tempfile.mkdtemp()
        self.dt.save(filepath)
        assert os.path.exists(f'{filepath}/dt.pkl')
        assert os.path.exists(f'{filepath}/dnn_nets.h5')
        newdt = deeptable.DeepTable.load(filepath)
        preds = newdt.predict(self.X_test)
        assert preds.shape, (200,)


if __name__ == "__main__":
    pass
