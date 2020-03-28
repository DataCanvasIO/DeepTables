# -*- coding:utf-8 -*-

import datetime
import gc
import os
import pprint
import time
import warnings
import copy
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, VerboseCallback

from . import dt_logging, consts, dart_early_stopping
from ..models import deeptable, modelset
from ..models.evaluation import calc_score

warnings.filterwarnings("ignore")
logger = dt_logging.get_logger()


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))


class BatchTrainer:
    def __init__(self,
                 data_train,
                 target,
                 data_test=None,
                 test_as_eval=False,
                 eval_size=0.2,
                 validation_size=0.2,
                 eval_metrics=[],
                 dt_config=None,
                 dt_nets=[['dnn_nets']],
                 dt_epochs=5,
                 dt_batch_size=128,
                 seed=9527,
                 pos_label=None,
                 verbose=1,
                 cross_validation=False,
                 retain_single_model=False,
                 num_folds=5,
                 stratified=True,
                 n_jobs=1,
                 lightgbm_params={},
                 catboost_params={},
                 ):
        self.verbose = verbose
        # logger.setLevel(self.verbose)

        if data_train is None:
            raise ValueError('[data_train] must be provided.')
        if dt_config is None:
            raise ValueError('[dt_config] must be provided.')
        if eval_metrics is None or len(eval_metrics) <= 0:
            raise ValueError('[eval_metrics] is empty, at least one metric.')

        if eval_size is not None and (eval_size >= 1.0 or eval_size < 0):
            raise ValueError(f'[eval_size] must be >= 0 and < 1.0.')

        if validation_size is not None and (validation_size >= 1.0 or validation_size < 0):
            raise ValueError(f'[validation_size] must be >= 0 and < 1.0.')

        if isinstance(dt_config, deeptable.ModelConfig):
            self.dt_config = [dt_config]
        elif isinstance(dt_config, list):
            self.dt_config = dt_config
        else:
            raise ValueError(f'[dt_config] can only be list or ModelConfig type.')

        self.dt_nets = dt_nets

        self.train = copy.deepcopy(data_train)
        self.test = data_test if data_test is None else copy.deepcopy(data_test)
        self.test_as_eval = test_as_eval
        self.target = target
        self.eval_size = eval_size
        self.validation_size = validation_size
        self.eval_metrics = eval_metrics
        self.dt_epochs = dt_epochs
        self.dt_batch_size = dt_batch_size
        self.seed = seed

        self.cross_validation = cross_validation
        self.retain_single_model = retain_single_model
        self.num_folds = num_folds
        self.stratified = stratified
        self.n_jobs = n_jobs
        self.lightgbm_params = lightgbm_params
        self.catboost_params = catboost_params
        self.__prepare_data()
        self.model_set = modelset.ModelSet(metric=self.first_metric_name, best_mode=consts.MODEL_SELECT_MODE_AUTO)

    @property
    def first_metric_name(self):
        if self.eval_metrics is None or len(self.eval_metrics)<=0:
            raise ValueError('`metrics` is none or empty.')
        first_metric = self.eval_metrics[0]
        if isinstance(first_metric, str):
            return first_metric
        if callable(first_metric):
            return first_metric.__name__
        raise ValueError('`metric` must be string or callable object.')

    def __prepare_data(self):
        if self.train is None:
            raise ValueError('Train set cannot be none.')
        if not isinstance(self.train, pd.DataFrame):
            self.train = pd.DataFrame(self.train)

        if self.train.columns.dtype != 'object':
            self.train.columns = ['x_' + str(c) for c in self.train.columns]

        if self.test_as_eval and self.test is not None:
            self.X_train = self.train
            self.X_eval = self.test
            self.test = None
        else:
            if self.eval_size > 0:
                self.X_train, self.X_eval = train_test_split(self.train, test_size=self.eval_size,
                                                             random_state=self.seed)
            else:
                self.X_train = self.train
                self.X_eval = None
                self.y_eval = None

        if self.test is not None:
            if not isinstance(self.test, pd.DataFrame):
                self.test = pd.DataFrame(self.test)
            if self.test.columns.dtype != 'object':
                self.test.columns = ['x_' + str(c) for c in self.test.columns]
            set_sub = set(self.train.columns) - set(self.test.columns)
            if len(set_sub) == 0:
                self.X_test = self.test
                self.y_test = self.X_test.pop(self.target)
            elif len(set_sub) == 1 and set_sub.pop() == self.target:
                self.X_test = self.test
                self.y_test = None
            else:
                raise ValueError(f'Train set and test set do not match.')
        else:
            self.X_test = None
            self.y_test = None

        self.y_train = self.X_train.pop(self.target).values
        self.y_eval = self.X_eval.pop(self.target).values if self.X_eval is not None else None
        self.task, labels = deeptable.infer_task_type(self.y_train)
        self.classes = len(labels)
        gc.collect()

    def train_catboost(self, config):
        return self.train_model(self.model_set, config, catboost_fit, 'CatBoost', **self.catboost_params)

    def train_lgbm(self, config):
        return self.train_model(self.model_set, config, lgbm_fit, 'LightGBM', **self.lightgbm_params)

    def start(self, models=['dt']):
        self.model_set = modelset.ModelSet(metric=self.first_metric_name, best_mode=consts.MODEL_SELECT_MODE_AUTO)
        for config in self.dt_config:
            if models is None or 'lightgbm' in models:
                with timer('LightGBM'):
                    self.train_lgbm(config)
                print('----------------------------------------------------------\n')
            if models is None or 'catboost' in models:
                with timer('CatBoost'):
                    self.train_catboost(config)
                print('----------------------------------------------------------\n')
            if models is None or 'dt' in models:
                for nets in self.dt_nets:
                    with timer(f'DT - {nets}'):
                        dt = self.train_dt(model_set=self.model_set, config=config, nets=nets)
                    print('----------------------------------------------------------\n')

        if models is None or 'autogluon' in models:
            with timer('AutoGluon'):
                print('')
            print('----------------------------------------------------------\n')

        if models is None or 'h2o' in models:
            with timer('H2O AutoML'):
                print('')
            print('----------------------------------------------------------\n')

        return self.model_set

    def train_dt(self, model_set, config, nets=['dnn_nets']):
        print(f'Start training DT model.{nets}')
        conf = config

        fixed_embedding_dim = conf.fixed_embedding_dim
        if 'fm_nets' in nets:
            fixed_embedding_dim = True
        print(f'train metrics:{config.metrics}')
        print(f'eval metrics:{self.eval_metrics}')
        # conf = conf._replace(nets=nets, metrics=[self.eval_metrics[0]],
        #                      fixed_embedding_dim=fixed_embedding_dim,
        #                      )

        dt = deeptable.DeepTable(config=conf)

        print(f'Fitting model...')
        if self.cross_validation:
            oof_proba, eval_proba, test_proba = dt.fit_cross_validation(self.X_train,
                                                                        self.y_train,
                                                                        self.X_eval,
                                                                        self.X_test,
                                                                        verbose=self.verbose,
                                                                        batch_size=self.dt_batch_size,
                                                                        epochs=self.dt_epochs,
                                                                        num_folds=self.num_folds,
                                                                        stratified=self.stratified,
                                                                        random_state=self.seed,
                                                                        n_jobs=self.n_jobs)
            print(f'Scoring...')
            oof_preds = dt.proba2predict(oof_proba)
            oof_score = calc_score(self.y_train, oof_proba, oof_preds, self.eval_metrics, self.task,
                                   dt.pos_label)
            model_set.push(
                modelset.ModelInfo('oof', f'{config.name} - {nets} - CV - oof', dt, oof_score,
                                   model_selector=consts.MODEL_SELECTOR_ALL))
            print(f'\n------------OOF------------ score:\n{oof_score}')

            if eval_proba is not None:
                eval_preds = dt.proba2predict(eval_proba)
                eval_cv_score = calc_score(self.y_eval, eval_proba, eval_preds, self.eval_metrics, self.task,
                                           dt.pos_label)
                model_set.push(
                    modelset.ModelInfo('cv-eval', f'{config.name} - {nets} - CV - eval', dt, eval_cv_score,
                                       model_selector=consts.MODEL_SELECTOR_ALL))
                print(f'\n------------CV------------ Eval score:\n{eval_cv_score}')

            if self.retain_single_model:
                all_model_proba = dt.predict_proba_all(self.X_eval)
                for fold_name, fold_proba in all_model_proba.items():
                    fold_preds = dt.proba2predict(fold_proba)
                    fold_score = calc_score(self.y_eval, fold_proba, fold_preds, self.eval_metrics, self.task,
                                            dt.pos_label)
                    print(f'\n------------{fold_name} -------------Eval score:\n{fold_score}')
                    model_set.push(
                        modelset.ModelInfo('eval', f'{config.name} - {nets} - {fold_name} - eval', dt, fold_score,
                                           model_selector=fold_name))

        else:
            print(f'X_train.shape:{self.X_train.shape},y_train.shape:{self.y_train.shape}')
            model, history = dt.fit(self.X_train,
                                    self.y_train,
                                    epochs=self.dt_epochs,
                                    validation_split=self.validation_size,
                                    verbose=self.verbose,
                                    )
            print(f'Scoring...')
            if self.X_eval is not None:
                proba = dt.predict_proba(self.X_eval, model_selector=consts.MODEL_SELECTOR_BEST)
                preds = dt.proba2predict(proba)
                score = calc_score(self.y_eval, proba, preds, self.eval_metrics, self.task, dt.pos_label)
                # score = dt.evaluate(self.X_test, self.y_test)
                print(f'\n------------{nets} -------------Eval score:\n{score}')
                model_set.push(
                    modelset.ModelInfo('eval', f'{config.name} - {nets} - eval', dt, score,
                                       model_selector=consts.MODEL_SELECTOR_BEST))
            else:
                print(f'\n------------{nets} -------------Val score:\n{history.history}')
                model_set.push(
                    modelset.ModelInfo('val', f'{config.name} - {nets} - val', dt, {},
                                       model_selector=consts.MODEL_SELECTOR_BEST,
                                       history=history.history))

            if self.X_test is not None:
                test_proba = dt.predict_proba(self.X_test)
                score = str(round(history.history[self.first_metric_name][-1], 5))
                file = f'{dt.output_path}{score}_{"_".join(nets)}.csv'
                pd.DataFrame(test_proba).to_csv(file, index=False)

        print(f'DT finished.')
        return dt

    def train_model(self, model_set, config, fit_fn, model_name, **params):
        model_name = f'{config.name}-{model_name}'
        print(f'Start training {model_name} model.')
        if params is None:
            params = {}
        if params.get('iterations') is None:
            params['iterations'] = 20

        conf = config
        conf = conf._replace(apply_gbm_features=False, auto_categorize=False)
        dt = deeptable.DeepTable(config=conf)
        print('Preparing datasets...')

        X_train, _ = dt.preprocessor.fit_transform(self.X_train, self.y_train)
        X, X_val, y, y_val = train_test_split(X_train, self.y_train, test_size=self.validation_size,
                                              random_state=self.seed)
        cat_vars = dt.preprocessor.get_categorical_columns()
        cont_vars = dt.preprocessor.get_continuous_columns()
        X[cat_vars] = X[cat_vars].astype('int')
        X[cont_vars] = X[cont_vars].astype('float')
        X_val[cat_vars] = X_val[cat_vars].astype('int')
        X_val[cont_vars] = X_val[cont_vars].astype('float')

        pos_label = dt.preprocessor.pos_label

        model = fit_fn(X, y,
                       X_val, y_val,
                       cat_vars,
                       self.task,
                       params)

        if self.X_eval is not None:
            X_eval = dt.preprocessor.transform_X(self.X_eval)
            X_eval[cat_vars] = X_eval[cat_vars].astype('int')
            X_eval[cont_vars] = X_eval[cont_vars].astype('float')

            y_eval = self.y_eval

            preds = model.predict(X_eval)

            if self.task == consts.TASK_REGRESSION:
                proba = preds
            elif self.task == consts.TASK_MULTICLASS:
                proba = model.predict_proba(X_eval)
            else:
                proba = model.predict_proba(X_eval)[:, 1]

            print('Scoring...')
            score = calc_score(self.y_eval, proba, preds, self.eval_metrics, self.task, pos_label)
            model_set.push(modelset.ModelInfo('eval', model_name, model, score, dt=dt))
            print(f'\n------------{model_name} -------------Eval score:\n{score}')
        else:
            print('No evaluation dataset specified.')
        print(f'{model_name} finished.')

        return model, score

    def get_models(self, models):
        if models is None or len(models) <= 0:
            raise ValueError(f'"models" is empty, at least 1 model.')

        if isinstance(models, str):
            if models == 'all':
                models = [mi.name for mi in self.model_set.get_modelinfos()]
            elif models.startswith('top'):
                n = int(models[3:])
                models = [mi.name for mi in self.model_set.top_n(n)]
            else:
                raise ValueError(f'"{models}" does not support.')
        mis = []
        for modelname in models:
            if isinstance(modelname, int):
                # modelname is a index
                mi = self.model_set.get_modelinfos()[modelname]
            else:
                mi = self.model_set.get_modelinfo(modelname)
            if mi is None:
                logger.warn(f'"{modelname}" not found in modelset.')
            else:
                mis.append(mi)
        return mis

    def gbm_model_predict_proba(self, dt, model, X):
        cat_vars = dt.preprocessor.get_categorical_columns()
        cont_vars = dt.preprocessor.get_continuous_columns()
        X = dt.preprocessor.transform_X(X)
        X[cat_vars] = X[cat_vars].astype('int')
        X[cont_vars] = X[cont_vars].astype('float')
        preds = model.predict(X)
        if self.task == consts.TASK_REGRESSION:
            proba = preds
        elif self.task == consts.TASK_MULTICLASS:
            proba = model.predict_proba(X)
        else:
            proba = model.predict_proba(X)[:, 1]
        return proba

    @staticmethod
    def fit_cross_validation(estimator_type,
                             fit_fn,
                             X,
                             y,
                             X_test=None,
                             score_fn=roc_auc_score,
                             estimator_params={},
                             categorical_feature=None,
                             task_type=consts.TASK_BINARY,
                             num_folds=5,
                             stratified=True,
                             iterators=None,
                             batch_size=None,
                             preds_filepath=None, ):
        print("Start cross validation")
        print(f'X.Shape={np.shape(X)}, y.Shape={np.shape(y)}, batch_size={batch_size}')

        # Cross validation model
        if iterators is None:
            if stratified:
                iterators = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=1001)
            else:
                iterators = KFold(n_splits=num_folds, shuffle=True, random_state=1001)
        print(f'Iterators:{iterators}')

        if len(y.shape) > 1:
            oof_proba = np.zeros(y.shape)
        else:
            oof_proba = np.zeros((y.shape[0], 1))

        y = np.array(y)
        if preds_filepath is None and os.environ.get(consts.ENV_DEEPTABLES_HOME) is not None:
            preds_filepath = os.environ.get(consts.ENV_DEEPTABLES_HOME)
        if preds_filepath is None:
            preds_filepath = f'./preds_{estimator_type}_{datetime.datetime.now().__format__("%Y_%m_%d %H:%M:%S")}/'

        if not os.path.exists(preds_filepath):
            os.makedirs(preds_filepath)

        for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(X, y)):
            print(f'\nFold:{n_fold + 1}\n')

            x_train_fold, y_train_fold = X.iloc[train_idx], y[train_idx]
            x_val_fold, y_val_fold = X.iloc[valid_idx], y[valid_idx]

            model = fit_fn(
                x_train_fold,
                y_train_fold,
                x_val_fold,
                y_val_fold,
                cat_vars=categorical_feature,
                task=task_type,
                estimator_params=estimator_params,
            )
            print(f'Fold {n_fold + 1} finished.')
            proba = model.predict_proba(x_val_fold)[:, 1:2]
            oof_proba[valid_idx] = proba
            test_fold_proba = model.predict_proba(X_test)
            score = round(score_fn(y_val_fold, proba), 5)
            file = f'{preds_filepath}{score}_fold{n_fold + 1}.csv'
            pd.DataFrame(test_fold_proba).to_csv(file, index=False)
            print(f'Fold {n_fold + 1} Score:{score}')

        if oof_proba.shape[-1] == 1:
            oof_proba = oof_proba.reshape(-1)
        print(f'OOF score:{score_fn(y, oof_proba)}')
        return oof_proba

    def ensemble_predict_proba(self, models, X=None, y=None, submission=None, submission_target='target'):
        mis = self.get_models(models)
        if X is None:
            if self.X_test is not None:
                X = self.X_test
                y = self.y_test
                print(f'Use [X_test] as X. X.shape:{X.shape}')
            else:
                X = self.X_eval
                y = self.y_eval
                print(f'Use [X_evel] as X. X.shape:{X.shape}')

        else:
            print(f'X.shape:{X.shape}')
        proba_avg = None
        count = 0
        score_dt = None
        X_dt = None
        for mi in mis:
            proba = None
            if isinstance(mi.model, deeptable.DeepTable):
                dt = mi.model
                if score_dt is None:
                    score_dt = dt
                model_selector = mi.meta.get('model_selector')
                if model_selector is None:
                    raise ValueError(f'Missing "model_selector" info.{mi.name}:{mi}')
                print(f'{mi.name} predicting...')
                if X_dt is None:
                    X_dt = dt.preprocessor.transform_X(X)
                proba = dt.predict_proba(X_dt, model_selector=model_selector, auto_transform_data=False)
            else:
                # perhaps be a gbm model
                gbm_model = mi.model
                dt = mi.meta.get('dt')
                if dt is None:
                    raise ValueError(f'Missing "dt" info.{mi.name}:{mi}')
                print(f'{mi.name} predicting...')
                proba = self.gbm_model_predict_proba(dt, gbm_model, X)
            if proba is not None:
                if len(proba.shape) == 1:
                    proba = proba.reshape((-1, 1))
                if proba_avg is None:
                    proba_avg = proba
                else:
                    proba_avg += proba
                count = count + 1
        proba_avg = proba_avg / count
        preds = None
        score = None
        if y is not None and score_dt is not None:
            preds = score_dt.proba2predict(proba_avg)
            score = calc_score(y, proba_avg, preds, self.eval_metrics, score_dt.task, score_dt.pos_label)
            print(f'\nEnsemble Test Score:\n{score}')
        if submission is not None and submission.shape[0] == proba_avg.shape[0]:
            submission[submission_target] = proba_avg[:, 0]
        return proba_avg[:, 0], preds, score, submission

    def probe_evaluate(self, models, layers, score_fn={}):
        mis = self.get_models(models)
        result = {}
        for mi in mis:
            proba = None
            if isinstance(mi.model, deeptable.DeepTable):
                dt = mi.model
                print(f'Evaluating...')
                score = deeptable.probe_evaluate(dt, self.X_train, self.y_train, self.X_eval, self.y_eval, layers,
                                                 score_fn)
                print(f'----------{mi.name}---------- Score:\n{score}')
                result[mi.name] = score
            else:
                print(f'Unsupported model type:{mi.name},{mi.model}')
        return result

    @staticmethod
    def hyperopt_search(optimizer, X, y, fit_params, title, callbacks=None):
        start = time.time()
        if isinstance(optimizer, BayesSearchCV):
            if fit_params is not None:
                optimizer.fit_params = fit_params
            optimizer.fit(X, y, callback=callbacks)
        else:
            optimizer.fit(X, y, **fit_params)

        d = pd.DataFrame(optimizer.cv_results_)
        best_score = optimizer.best_score_
        best_score_std = d.iloc[optimizer.best_index_].std_test_score
        best_params = optimizer.best_params_
        print(f"{title} took {time.time() - start} seconds,  "
              f"candidates checked: {len(optimizer.cv_results_['params'])}, "
              f"best CV score: {best_score} "
              f'\u00B1 {best_score_std}')
        print('Best parameters:')
        pprint.pprint(best_params)
        return best_params

    @staticmethod
    def grid_search(estimator, param_grid, X, y,
                    fit_params=None, scoring=None, n_jobs=None, cv=None, refit=False, verbose=0):
        optimizer = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv,
            refit=refit,
            verbose=verbose)
        best_parmas = BatchTrainer.hyperopt_search(optimizer, X, y, fit_params=fit_params, title='GridSearchCV')
        return best_parmas

    @staticmethod
    def randomized_search(estimator, param_distributions, X, y,
                          fit_params=None, scoring=None, n_jobs=None, cv=None, n_iter=10, refit=False, verbose=0):
        optimizer = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            scoring=scoring,
            n_jobs=n_jobs,
            cv=cv,
            n_iter=n_iter,
            refit=refit,
            verbose=verbose)
        best_parmas = BatchTrainer.hyperopt_search(optimizer, X, y, fit_params=fit_params, title='RandomizedSearchCV')
        return best_parmas

    @staticmethod
    def bayes_search(
            estimator, search_spaces, X, y,
            fit_params=None, scoring=None, n_jobs=1, cv=None, n_points=1,
            n_iter=50, refit=False, random_state=9527, verbose=0, deadline=60):
        optimizer = BayesSearchCV(
            estimator,
            search_spaces,
            scoring=scoring,
            cv=cv,
            n_points=n_points,
            n_iter=n_iter,
            n_jobs=n_jobs,
            return_train_score=False,
            refit=refit,
            optimizer_kwargs={'base_estimator': 'GP'},
            random_state=random_state)
        best_parmas = BatchTrainer.hyperopt_search(
            optimizer, X, y, fit_params=fit_params, title='BayesSearchCV',
            callbacks=[VerboseCallback(verbose), DeadlineStopper(deadline)])
        return best_parmas


def catboost_fit(X, y, X_val, y_val, cat_vars, task, estimator_params):
    from catboost import CatBoostClassifier, CatBoostRegressor
    if task == consts.TASK_REGRESSION:
        catboost = CatBoostRegressor(**estimator_params)
    else:
        catboost = CatBoostClassifier(**estimator_params)

    print('Fitting model...')

    catboost.fit(X, y,
                 # eval_metric=self.metrics[0],
                 eval_set=[(X_val, y_val)],
                 # verbose=100,
                 early_stopping_rounds=200,
                 cat_features=cat_vars)
    print('Scoring...')

    return catboost


def lgbm_fit(X, y, X_val, y_val, cat_vars, task, estimator_params):
    from lightgbm import LGBMClassifier, LGBMRegressor

    if task == consts.TASK_REGRESSION:
        lgbm = LGBMRegressor(**estimator_params)
    else:
        lgbm = LGBMClassifier(**estimator_params)

    print('Fitting model...')
    callback = None
    if estimator_params.get('boosting') == 'dart':
        callback = [dart_early_stopping.dart_early_stopping(200, first_metric_only=True)]

    lgbm.fit(X, y,
             # eval_metric=self.metrics[0],
             eval_set=[(X_val, y_val)],
             verbose=100,
             callbacks=callback,
             early_stopping_rounds=200,
             categorical_feature=cat_vars)
    print('Predicting...')

    return lgbm
