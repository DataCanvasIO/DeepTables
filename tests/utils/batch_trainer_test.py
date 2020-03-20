# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from deeptables.utils import batch_trainer
from deeptables.datasets import dsutils
from deeptables.models import deeptable
from sklearn.datasets import load_boston
import pandas as pd


class Test_Batch_Trainer:
    def test_run_binary_heart_disease_CV(self):
        data = dsutils.load_heart_disease_uci()
        conf = deeptable.ModelConfig(
            dnn_params={'dnn_units': ((256, 0, False), (256, 0, False)),
                        'dnn_activation': 'relu'},
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            auto_discrete=True,
            auto_categorization=False,
            cat_exponent=0.4,
            cat_remain_numeric=True,
            # optimizer=keras.optimizers.RMSprop(),
            monitor_metric='val_loss',
        )
        bt = batch_trainer.BatchTrainer(data, 'target',
                                        eval_size=0.2,
                                        validation_size=0.2,
                                        metrics=['AUC', 'accuracy', 'recall', 'precision', 'f1'],
                                        # AUC/recall/precision/f1/mse/mae/msle/rmse/r2
                                        dt_config=conf,
                                        verbose=0,
                                        dt_epochs=1,
                                        cross_validation=True,
                                        num_folds=3,
                                        # seed=9527,
                                        )
        ms = bt.start()
        assert ms.leaderboard().shape[1], 7

    def test_run_lgbm(self):
        data = dsutils.load_adult().head(1000)
        conf = deeptable.ModelConfig(
            dnn_params={'dnn_units': ((256, 0, False), (256, 0, False)),
                        'dnn_activation': 'relu'},
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            # auto_discrete=True,
        )
        bt = batch_trainer.BatchTrainer(data, 'x_14',
                                        eval_size=0.2,
                                        validation_size=0.2,
                                        metrics=['AUC', 'accuracy', 'recall', 'precision', 'f1'],
                                        # AUC/recall/precision/f1/mse/mae/msle/rmse/r2
                                        dt_config=conf,
                                        verbose=0,
                                        dt_epochs=1,
                                        # seed=9527,
                                        lightgbm_params={'learning_rate': 0.01, 'colsample_bytree': 0.95,
                                                         'reg_alpha': 0.04, 'reg_lambda': 0.07},
                                        )
        lgbm, score = bt.train_lgbm(conf)
        assert lgbm
        assert score['auc'] > 0

    def test_run_catboost(self):
        data = dsutils.load_adult().head(1000)
        conf = deeptable.ModelConfig(
            dnn_params={'dnn_units': ((256, 0, False), (256, 0, False)),
                        'dnn_activation': 'relu'},
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            # auto_discrete=True,
        )
        bt = batch_trainer.BatchTrainer(data, 'x_14',
                                        eval_size=0.2,
                                        validation_size=0.2,
                                        metrics=['AUC', 'accuracy', 'recall', 'precision', 'f1'],
                                        # AUC/recall/precision/f1/mse/mae/msle/rmse/r2
                                        dt_config=conf,
                                        verbose=0,
                                        dt_epochs=1,
                                        catboost_params={'iterations': 5}
                                        # seed=9527,
                                        )
        cb, score = bt.train_catboost(conf)
        assert cb
        assert score['auc'] > 0

    def test_run_binary(self):
        data = dsutils.load_adult().head(1000)
        conf = deeptable.ModelConfig(
            dnn_params={'dnn_units': ((256, 0, False), (256, 0, False)),
                        'dnn_activation': 'relu'},
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            # auto_discrete=True,
        )
        bt = batch_trainer.BatchTrainer(data, 'x_14',
                                        eval_size=0.2,
                                        validation_size=0.2,
                                        metrics=['AUC', 'accuracy', 'recall', 'precision', 'f1'],
                                        # AUC/recall/precision/f1/mse/mae/msle/rmse/r2
                                        dt_config=conf,
                                        verbose=0,
                                        dt_epochs=1,
                                        # seed=9527,
                                        )
        ms = bt.start()
        assert ms.leaderboard().shape[1], 7

    def test_run_regression(self):
        boston_dataset = load_boston()
        df_train = pd.DataFrame(boston_dataset.data)
        df_train.columns = boston_dataset.feature_names
        df_train.insert(df_train.shape[1], 'target', boston_dataset.target)

        conf = deeptable.ModelConfig(
            dnn_params={'dnn_units': ((256, 0, False), (256, 0, False)),
                        'dnn_activation': 'relu'},
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=True,
            auto_discrete=False,
            auto_categorization=False,
            cat_exponent=0.2,
            cat_remain_numeric=True,
        )

        bt = batch_trainer.BatchTrainer(df_train, 'target',
                                        eval_size=0.2,
                                        validation_size=0.2,
                                        # metrics=['AUC','accuracy','recall','precision','f1'], #auc/recall/precision/f1/mse/mae/msle/rmse/r2
                                        metrics=['mse', 'rmse', 'r2', 'mae'],
                                        # auc/recall/precision/f1/mse/mae/msle/rmse/r2
                                        verbose=0,
                                        dt_config=conf,
                                        dt_nets=[['dnn_nets'], ['dnn_nets', 'cross_nets'], ['cross_nets']],
                                        dt_epochs=1,
                                        # seed=9527,
                                        )

        ms = bt.start()
        assert ms.leaderboard().shape[1], 6

    def test_run_multiclass(self):
        data = dsutils.load_glass_uci()
        conf = deeptable.ModelConfig(
            # dnn_units=((256, 0, False), (128, 0, False)),
            # dnn_activation='relu',
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            auto_discrete=False,
        )
        bt = batch_trainer.BatchTrainer(data, 'x_10',
                                        eval_size=0.2,
                                        validation_size=0.2,
                                        metrics=['AUC', 'accuracy', 'recall', 'precision', 'f1'],
                                        # AUC/recall/precision/f1/mse/mae/msle/rmse/r2
                                        dt_config=conf,
                                        verbose=0,
                                        dt_epochs=1,
                                        # seed=9527,
                                        cross_validation=True,
                                        stratified=False,
                                        num_folds=5,
                                        )
        ms = bt.start()
        assert ms.leaderboard().shape[1], 7

    def test_run_cross_validation(self):
        data = dsutils.load_adult().head(1000)
        conf = deeptable.ModelConfig(
            # dnn_units=((256, 0, False), (128, 0, False)),
            # dnn_activation='relu',
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            auto_discrete=False,
        )
        bt = batch_trainer.BatchTrainer(data, 'x_14',
                                        data_test=data,
                                        eval_size=0.2,
                                        validation_size=0.2,
                                        metrics=['AUC', 'accuracy', 'recall', 'precision', 'f1'],
                                        # AUC/recall/precision/f1/mse/mae/msle/rmse/r2
                                        dt_config=conf,
                                        verbose=0,
                                        dt_epochs=1,
                                        # seed=9527,
                                        cross_validation=True,
                                        num_folds=5,
                                        )
        ms = bt.start(models=['dt'])
        assert ms.leaderboard().shape[1], 7

    def test_get_models(self):
        data = dsutils.load_adult().head(1000)
        conf = deeptable.ModelConfig(
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            auto_discrete=False,
        )
        bt = batch_trainer.BatchTrainer(data, 'x_14',
                                        eval_size=0.2,
                                        validation_size=0.2,
                                        metrics=['AUC', 'accuracy', 'recall', 'precision', 'f1'],
                                        # AUC/recall/precision/f1/mse/mae/msle/rmse/r2
                                        dt_config=conf,
                                        verbose=0,
                                        dt_epochs=1,
                                        # seed=9527,
                                        cross_validation=True,
                                        num_folds=5,
                                        )
        ms = bt.start(models=None)
        mis_all = bt.get_models('all')
        mis_top2 = bt.get_models('top2')
        mis_modelindex = bt.get_models([1, 3])
        mis_modelnames = bt.get_models(['conf-1 - [\'dnn_nets\'] - CV - oof',
                                        'conf-1 - [\'dnn_nets\'] - CV - eval',
                                        'LightGBM',
                                        'CatBoost'])

        assert len(mis_all), 4
        assert len(mis_top2), 2
        assert len(mis_modelindex), 2
        assert len(mis_modelnames), 4
        assert mis_modelnames[0].name, 'conf-1 - [\'dnn_nets\'] - CV - oof'

    def test_get_models_retian_single_model(self):
        data = dsutils.load_adult().head(1000)
        conf = deeptable.ModelConfig(
            # dnn_units=((256, 0, False), (128, 0, False)),
            # dnn_activation='relu',
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            auto_discrete=False,
        )
        bt = batch_trainer.BatchTrainer(data, 'x_14',
                                        eval_size=0.2,
                                        validation_size=0.2,
                                        metrics=['AUC', 'accuracy', 'recall', 'precision', 'f1'],
                                        # AUC/recall/precision/f1/mse/mae/msle/rmse/r2
                                        dt_config=conf,
                                        verbose=0,
                                        dt_epochs=1,
                                        # seed=9527,
                                        cross_validation=True,
                                        num_folds=2,
                                        retain_single_model=True,
                                        )
        ms = bt.start()
        mis_all = bt.get_models('all')
        mis_top2 = bt.get_models('top2')
        mis_modelindex = bt.get_models([1, 3])
        mis_modelnames = bt.get_models(['conf-1 - [\'dnn_nets\'] - CV - oof',
                                        'conf-1 - [\'dnn_nets\'] - dnn_nets-kfold-1 - eval',
                                        'LightGBM',
                                        'CatBoost'])
        assert len(mis_all), 6
        assert len(mis_top2), 2
        assert len(mis_modelnames), 4
        assert len(mis_modelindex), 2
        assert mis_modelnames[0].name, 'conf-1 - [\'dnn_nets\'] - CV - oof'

    def test_ensemble_predict_proba(self):
        data = dsutils.load_adult().head(1000)
        conf = deeptable.ModelConfig(
            # dnn_units=((256, 0, False), (128, 0, False)),
            # dnn_activation='relu',
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            auto_discrete=False,
        )
        bt = batch_trainer.BatchTrainer(data, 'x_14',
                                        eval_size=0.2,
                                        validation_size=0.2,
                                        metrics=['AUC', 'accuracy', 'recall', 'precision', 'f1'],
                                        # AUC/recall/precision/f1/mse/mae/msle/rmse/r2
                                        dt_config=conf,
                                        verbose=0,
                                        dt_epochs=1,
                                        # seed=9527,
                                        cross_validation=True,
                                        num_folds=5,
                                        )
        ms = bt.start()
        proba, preds, score, submission = bt.ensemble_predict_proba('all')

        assert proba.shape, (6513,)

    def test_probe_evaluation(self):
        data = dsutils.load_adult().head(1000)
        conf = deeptable.ModelConfig(
            # dnn_units=((256, 0, False), (128, 0, False)),
            # dnn_activation='relu',
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            auto_discrete=False,
        )
        bt = batch_trainer.BatchTrainer(data, 'x_14',
                                        eval_size=0.2,
                                        validation_size=0.2,
                                        metrics=['AUC', 'accuracy', 'recall', 'precision', 'f1'],
                                        dt_config=conf,
                                        verbose=0,
                                        dt_epochs=1,
                                        cross_validation=False,
                                        )
        ms = bt.start(models=['dt'])
        result = bt.probe_evaluate('all', layers=['flatten_embeddings', 'dnn_dense_1', 'dnn_dense_2'])
        assert len(result), 1
        assert len(result["conf-1 - ['dnn_nets'] - eval"]), 3

    def test_zero_testset(self):
        data = dsutils.load_adult().head(1000)
        conf = deeptable.ModelConfig(
            # dnn_units=((256, 0, False), (128, 0, False)),
            # dnn_activation='relu',
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            auto_discrete=False,
        )
        bt = batch_trainer.BatchTrainer(data, 'x_14',
                                        eval_size=0,
                                        validation_size=0.2,
                                        metrics=['AUC', 'accuracy', 'recall', 'precision', 'f1'],
                                        dt_config=conf,
                                        verbose=0,
                                        dt_epochs=1,
                                        cross_validation=False,
                                        )
        assert len(bt.X_train), 1000
        assert bt.X_eval is None

        ms = bt.start(models=['dt'])
        assert len(ms.get_models()), 1

    def test_zero_testset_cross_validation(self):
        data = dsutils.load_adult().head(1000)
        conf = deeptable.ModelConfig(
            # dnn_units=((256, 0, False), (128, 0, False)),
            # dnn_activation='relu',
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            auto_discrete=False,
        )
        bt = batch_trainer.BatchTrainer(data, 'x_14',
                                        eval_size=0,
                                        validation_size=0.2,
                                        metrics=['AUC', 'accuracy', 'recall', 'precision', 'f1'],
                                        dt_config=conf,
                                        verbose=0,
                                        dt_epochs=1,
                                        cross_validation=True,
                                        num_folds=2,
                                        retain_single_model=False,
                                        )
        assert len(bt.X_train), 1000
        assert bt.X_eval is None

        ms = bt.start(models=['dt'])
        assert len(ms.get_models()), 1

    def test_multi_config(self):
        data = dsutils.load_adult().head(1000)
        conf1 = deeptable.ModelConfig(
            name='conf001',
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            auto_discrete=False,
        )
        conf2 = deeptable.ModelConfig(
            name='conf002',
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            auto_discrete=False,
        )
        bt = batch_trainer.BatchTrainer(data, 'x_14',
                                        eval_size=0,
                                        validation_size=0.2,
                                        metrics=['AUC', 'accuracy', 'recall', 'precision', 'f1'],
                                        dt_config=[conf1,conf2],
                                        verbose=0,
                                        dt_epochs=1,
                                        cross_validation=True,
                                        num_folds=2,
                                        retain_single_model=False,
                                        )

        ms = bt.start(models=['dt'])
        assert len(ms.get_models()), 2


    def test_leaderboard(self):
        data = dsutils.load_adult().head(1000)
        conf = deeptable.ModelConfig(
            # dnn_units=((256, 0, False), (128, 0, False)),
            # dnn_activation='relu',
            fixed_embedding_dim=False,
            embeddings_output_dim=0,
            apply_gbm_features=False,
            auto_discrete=False,
        )
        bt = batch_trainer.BatchTrainer(data, 'x_14',
                                        eval_size=0.2,
                                        validation_size=0.2,
                                        metrics=['AUC', 'accuracy', 'recall', 'precision', 'f1'],
                                        dt_config=conf,
                                        verbose=0,
                                        dt_epochs=1,
                                        cross_validation=True,
                                        num_folds=2,
                                        retain_single_model=True,
                                        )

        ms = bt.start()
        eval_lb = ms.leaderboard(type='eval')
        oof_lb = ms.leaderboard(type='oof')
        val_lb = ms.leaderboard(type='val')
        assert len(eval_lb), 5
        assert len(oof_lb), 1
        assert val_lb is None
