# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import random
import time

import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn.metrics import roc_auc_score, mean_squared_error, f1_score

from deeptables.datasets import dsutils
from deeptables.models import deeptable, deepmodel
from deeptables.utils import consts, fs
from hypernets.tabular import get_tool_box
from hypernets.tests.tabular.tb_dask import is_dask_installed, if_dask_ready, setup_dask

if is_dask_installed:
    import dask.dataframe as dd


@if_dask_ready
class Test_DeepTable_Dask:
    @classmethod
    def setup_class(cls):
        setup_dask(cls)

        print("Loading datasets...")
        row_count = 1000
        df = dsutils.load_adult().head(row_count)

        cls.df = dd.from_pandas(df, npartitions=2)
        cls.df_row_count = row_count
        cls.target = 14

        print(f'Class {cls.__name__} setup.')

    @classmethod
    def teardown_class(cls):
        print("Class teardown.")

    def run_dt(self, config, df=None, target=None, fit_kwargs={}):
        if df is None or target is None:
            df = self.df.copy()
            target = self.target

        X_train, X_test = get_tool_box(df).train_test_split(df, test_size=0.2, random_state=9527)
        y_train = X_train.pop(target)
        y_test = X_test.pop(target)
        test_size = len(X_test)

        dt = deeptable.DeepTable(config=config)

        if fit_kwargs is None:
            fit_kwargs = {'epochs': 1}
        else:
            fit_kwargs = {'epochs': 1, **fit_kwargs}

        dm, history = dt.fit(X_train, y_train, **fit_kwargs)
        assert dm is not None
        assert history is not None

        # test evaluate
        result = dt.evaluate(X_test, y_test)
        assert result.get(config.metrics[0]) is not None
        print('evaluate:', result)

        # test_model_selector(self):
        m1 = dt.get_model(consts.MODEL_SELECTOR_CURRENT)
        m2 = dt.get_model(consts.MODEL_SELECTOR_BEST)
        m3 = dt.get_model('dnn_nets')

        assert isinstance(m1, deepmodel.DeepModel)
        assert m1 is m2
        assert m2 is m3

        # test_best_model(self):
        model = dt.best_model
        assert isinstance(model, deepmodel.DeepModel)

        if dt.task in [consts.TASK_BINARY, consts.TASK_MULTICLASS]:
            # test_predict_proba(self):
            num_classes = dt.num_classes
            proba = dt.predict_proba(X_test)

            assert proba.shape == (test_size, num_classes)

            # test_proba2predict(self):
            proba = dt.predict_proba(X_test)
            preds = dt.predict(X_test)
            preds2 = dt.proba2predict(proba)
            assert proba.shape == (test_size, num_classes)
            assert (preds == preds2).sum() == test_size
            assert preds2.shape == (test_size,)
        elif dt.task in [consts.TASK_REGRESSION, ]:
            preds = dt.predict(X_test)
            assert preds.shape == (test_size, 1)

        # test_apply(self):
        features = dt.apply(X_test,
                            output_layers=['flatten_embeddings', 'dnn_dense_1', 'dnn_dense_2'])
        assert len(features) == 3
        assert len(features[0].shape) == 2

        features = dt.apply(X_test, output_layers=['flatten_embeddings'])
        assert len(features.shape) == 2

        #  test_apply_with_transformer(self):
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)

        out1 = random.sample(range(test_size), test_size // 2)
        # X_sample = X_test.iloc[out1,]
        X_test_values = X_test.to_dask_array(lengths=True)
        samples = get_tool_box(X_test).make_chunk_size_known(X_test_values[out1])
        X_sample = dd.from_array(samples, columns=X_test.columns)

        features = dt.apply(X_sample,
                            output_layers=['flatten_embeddings', 'dnn_dense_1'],
                            transformer=tsne)
        assert len(features) == 2
        assert len(features[0].shape) == 2
        assert features[0].shape[1] == 2
        assert features[1].shape[1] == 2

        # def test_probe_evaluate(self):
        result = deeptable.probe_evaluate(dt, X_train, y_train, X_test, y_test,
                                          layers=['flatten_embeddings'], score_fn={})

        assert len(result) == 1
        assert result['flatten_embeddings']['accuracy'] > 0

        scores = {'MSE': mean_squared_error}
        if dt.task in [consts.TASK_BINARY, consts.TASK_MULTICLASS]:
            scores = {'AUC': roc_auc_score, 'F1': f1_score, **scores}
        result = deeptable.probe_evaluate(dt, X_train, y_train, X_test, y_test,
                                          layers=['flatten_embeddings', 'dnn_dense_1', 'dnn_dense_2'],
                                          score_fn=scores)

        assert len(result) == 3
        assert len(result['flatten_embeddings']) == len(scores)
        if dt.task in [consts.TASK_BINARY, consts.TASK_MULTICLASS]:
            assert result['flatten_embeddings']['AUC'] > 0
            assert result['dnn_dense_2']['AUC'] > 0

        return dt, dm

    def test_default_settings(self):
        config = deeptable.ModelConfig(metrics=['AUC'], apply_gbm_features=False, apply_class_weight=True)
        dt, _ = self.run_dt(config)

        # test save and load
        filepath = f'{type(self).__name__}_{time.strftime("%Y%m%d%H%M%S")}'
        dt.save(filepath)
        assert fs.exists(f'{filepath}/dt.pkl')
        assert fs.exists(f'{filepath}/dnn_nets.h5')
        newdt = deeptable.DeepTable.load(filepath)
        X_eval = self.df.copy()
        X_eval.pop(self.target)
        preds = newdt.predict(X_eval)
        assert preds.shape, (self.df_row_count,)

    def test_var_len_encoder(self):
        df = dd.from_pandas(dsutils.load_movielens(), npartitions=2)

        config = deeptable.ModelConfig(nets=['dnn_nets'],
                                       task=consts.TASK_REGRESSION,
                                       categorical_columns=["movie_id", "user_id", "gender", "occupation", "zip",
                                                            "title", "age"],
                                       metrics=['mse'],
                                       fixed_embedding_dim=True,
                                       embeddings_output_dim=4,
                                       apply_gbm_features=False,
                                       apply_class_weight=True,
                                       earlystopping_patience=5,
                                       var_len_categorical_columns=[('genres', "|", "max")])

        self.run_dt(config, df=df, target='rating')

    def test_gbm_features(self):
        config = deeptable.ModelConfig(metrics=['AUC'], apply_gbm_features=True, apply_class_weight=True)
        self.run_dt(config)

    def test_gbm_features_with_params(self):
        params = {'learning_rate': 0.01, 'colsample_bytree': 0.95,
                  'reg_alpha': 0.04, 'reg_lambda': 0.07,
                  'n_estimators': 10}
        config = deeptable.ModelConfig(metrics=['AUC'],
                                       apply_gbm_features=True,
                                       gbm_params=params,
                                       )
        dt, _ = self.run_dt(config)
        lgbm = dt.preprocessor.X_transformers['gbm_features'].lgbm
        assert all([getattr(lgbm, k, None) == v for k, v in params.items()])

        lgbm_leaves = [c for c in dt.preprocessor.get_categorical_columns() if 'lgbm_leaf' in c]
        assert len(lgbm_leaves), 10

    def test_gbm_feature_dense(self):
        params = {'learning_rate': 0.01, 'colsample_bytree': 0.95,
                  'reg_alpha': 0.04, 'reg_lambda': 0.07,
                  'n_estimators': 10}
        config = deeptable.ModelConfig(metrics=['AUC'],
                                       apply_gbm_features=True,
                                       gbm_feature_type=consts.GBM_FEATURE_TYPE_DENSE,
                                       gbm_params=params,
                                       )
        dt, dm = self.run_dt(config)
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

        df = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'y': y})
        df = dd.from_pandas(df, npartitions=1)
        y = df.pop('y')

        dt = deeptable.DeepTable(config=deeptable.ModelConfig(
            apply_gbm_features=False,
            auto_categorize=True,
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
        dft = dd.from_pandas(dft, npartitions=2)
        preds = dt.predict(dft)
        assert len(preds), 10


if __name__ == "__main__":
    pass
