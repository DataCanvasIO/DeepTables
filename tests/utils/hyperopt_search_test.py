# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from deeptables.utils.batch_trainer import BatchTrainer
from deeptables.utils.batch_trainer import lgbm_fit
from deeptables.datasets import dsutils
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Skopt functions
from skopt.space import Real, Integer
from scipy.stats import uniform as sp_uniform
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class hyperopt_search:
    def test_opt_catboost(self):
        df_train = dsutils.load_adult().head(1000)
        y = df_train.pop(14).values
        X = df_train
        cols = X.columns
        num_cols = X._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))
        clf = CatBoostClassifier(thread_count=4,
                                 loss_function='Logloss',
                                 cat_features=cat_cols,
                                 od_type='Iter',
                                 nan_mode='Min',
                                 iterations=1,
                                 eval_metric='AUC',
                                 metric_period=50,
                                 verbose=False
                                 )
        fit_params = {'early_stopping_rounds': 10}
        # randomized_search
        param_distributions = {
            # 'iterations': sp_randint(10, 1000),
            'depth': [1, 3, 5],  # sp_randint(1, 5),
            'learning_rate': sp_uniform(0.01, 1.0),

        }
        best_params1 = BatchTrainer.randomized_search(
            clf, param_distributions, X, y,
            fit_params=fit_params, scoring='roc_auc', n_jobs=1, cv=5)

        # grid_search
        param_grid = {
            # 'iterations': [10, 30],
            'depth': [1, 3, 5],  # sp_randint(1, 5),
            'learning_rate': [0.01, 0.05, 0.1],
        }
        best_params2 = BatchTrainer.grid_search(
            clf, param_grid, X, y,
            fit_params=fit_params, scoring='roc_auc', n_jobs=1, cv=5)

        # bayes_search
        search_spaces = {
            'depth': Integer(1, 5),
            'learning_rate': Real(0.02, 0.6, 'log-uniform'),
        }
        best_params3 = BatchTrainer.bayes_search(
            clf, search_spaces, X, y,
            fit_params=fit_params, scoring='roc_auc', n_jobs=1, cv=5, n_iter=10)

        assert best_params1['depth'] > 0
        assert best_params2['depth'] > 0
        assert best_params3['depth'] > 0

    def test_opt_lightgbm(self):
        df_train = dsutils.load_adult().head(1000)
        y = df_train.pop(14).values
        X = df_train
        cols = X.columns
        num_cols = X._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))

        le = LabelEncoder()
        for c in cat_cols:
            X[c] = le.fit_transform(X[c])

        clf = LGBMClassifier(
            n_estimators=10,
            boosting_type='gbdt',
            categorical_feature=cat_cols,
            num_leaves=31)
        fit_params = {'eval_metric': 'roc_auc'}
        # randomized_search
        param_distributions = {
            # 'iterations': sp_randint(10, 1000),
            'max_depth': [1, 3, 5],  # sp_randint(1, 5),
            'learning_rate': sp_uniform(0.01, 1.0),
        }
        best_params1 = BatchTrainer.randomized_search(
            clf, param_distributions, X, y,
            fit_params=fit_params, scoring='roc_auc', n_jobs=1, cv=5)

        # grid_search
        param_grid = {
            # 'iterations': [10, 30],
            'max_depth': [1, 3, 5],  # sp_randint(1, 5),
            'learning_rate': [0.01, 0.05, 0.1],
        }
        best_params2 = BatchTrainer.grid_search(
            clf, param_grid, X, y,
            fit_params=fit_params, scoring='roc_auc', n_jobs=1, cv=5)

        # bayes_search
        search_spaces = {
            'max_depth': Integer(1, 5),
            'learning_rate': Real(0.02, 0.6, 'log-uniform'),
        }
        best_params3 = BatchTrainer.bayes_search(
            clf, search_spaces, X, y,
            fit_params=fit_params, scoring='roc_auc', n_jobs=1, cv=5, n_iter=10)

        assert best_params1['max_depth'] > 0
        assert best_params2['max_depth'] > 0
        assert best_params3['max_depth'] > 0

    def test_fit_cv(self):
        df_train = dsutils.load_adult().head(1000)

        y = df_train.pop(14).values
        X = df_train
        cols = X.columns
        num_cols = X._get_numeric_data().columns
        cat_cols = list(set(cols) - set(num_cols))

        le = LabelEncoder()
        for c in cat_cols:
            X[c] = le.fit_transform(X[c])
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

        oof_proba = BatchTrainer.fit_cross_validation('lightGBM',
                                                      lgbm_fit,
                                                      X_train,
                                                      y_train,
                                                      X_test,
                                                      score_fn=roc_auc_score,
                                                      estimator_params={'max_depth': 3, 'learning_rate': 0.01},
                                                      categorical_feature=cols,
                                                      task_type='binary',
                                                      num_folds=5,
                                                      stratified=True,
                                                      iterators=None,
                                                      batch_size=None,
                                                      preds_filepath=None,
                                                      )
        auc = roc_auc_score(y_train, oof_proba)
        assert auc > 0
