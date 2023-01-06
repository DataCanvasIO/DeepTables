# -*- coding:utf-8 -*-
"""Training and inference for tabular datasets using neural nets."""

import os
import pickle
import time

import dask
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate, BatchNormalization

from hypernets.tabular import get_tool_box, is_dask_installed
from . import modelset, deepnets
from .config import ModelConfig
from .deepmodel import DeepModel
from .preprocessor import DefaultPreprocessor, DefaultDaskPreprocessor
from ..utils import dt_logging, consts, fs
from ..utils.tf_version import tf_less_than

if is_dask_installed:
    from hypernets.tabular.dask_ex import DaskToolBox

logger = dt_logging.get_logger(__name__)


class DeepTable:
    """`DeepTables` can be use to solve classification and regression prediction problems on tabular datasets.
    Easy to use and provide good performance out of box, no datasets preprocessing is required.

    Arguments
    ---------
    config : ModelConfig

        Options of ModelConfig
        ----------------------
            name: str, (default='conf-1')

            nets: list of str or callable object, (default=['dnn_nets'])
                Preset Nets
                -----------
                - DeepFM    -> ['linear','dnn_nets','fm_nets']
                - xDeepFM
                - DCN
                - PNN
                - WideDeep
                - AutoInt
                - AFM
                - FGCNN
                - FibiNet

                Avalible Build Blocks
                ---------------------
                - 'dnn_nets'
                - 'linear'
                - 'cin_nets'
                - 'fm_nets'
                - 'afm_nets'
                - 'opnn_nets'
                - 'ipnn_nets'
                - 'pnn_nets',
                - 'cross_nets'
                - 'cross_dnn_nets'
                - 'dcn_nets',
                - 'autoint_nets'
                - 'fg_nets'
                - 'fgcnn_cin_nets'
                - 'fgcnn_fm_nets'
                - 'fgcnn_ipnn_nets'
                - 'fgcnn_dnn_nets'
                - 'fibi_nets'
                - 'fibi_dnn_nets'

                Examples
                --------
                >>>from deeptables.models import deepnets
                >>>#preset nets
                >>>conf = ModelConfig(nets=deepnets.DeepFM)
                >>>#list of names of nets
                >>>conf = ModelConfig(nets=['linear','dnn_nets','cin_nets','cross_nets'])
                >>>#mixed preset nets and names
                >>>conf = ModelConfig(nets=deepnets.WideDeep+['cin_nets'])
                >>>#mixed names and custom nets
                >>>def custom_net(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
                >>>     out = layers.Dense(10)(flatten_emb_layer)
                >>>     return out
                >>>conf = ModelConfig(nets=['linear', custom_net])

            categorical_columns: list of strings, (default='auto')
                - 'auto'
                    get the columns of categorical type automatically. By default, the object,
                    bool and category will be selected.
                    if 'auto' the [auto_categorize] will no longer takes effect.
                - list of strings
                    e.g. ['x1','x2','x3','..']

            exclude_columns: list of strings, (default=[])

            pos_label: str or int, (default=None)
                The label of positive class, used only when task is binary.

            metrics: list of string or callable object, (default=['accuracy'])
                List of metrics to be evaluated by the model during training and testing.
                Typically, you will use `metrics=['accuracy']` or `metrics=['AUC']`.
                Every metric should be a built-in evaluation metric in tf.keras.metrics or a tf.keras.metrics Object
                or a callable object like `r2(y_true, y_pred):...` .
                See also: https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/keras/metrics

            auto_categorize: bool, (default=False)

            cat_exponent: float, (default=0.5)

            cat_remain_numeric: bool, (default=True)

            auto_encode_label: bool, (default=True)

            auto_imputation: bool, (default=True)

            auto_discrete: bool, (default=False)

            apply_gbm_features: bool, (default=False)

            gbm_params: dict, (default={})

            gbm_feature_type: str, (default=embedding)
                - embedding
                - dense

            fixed_embedding_dim: bool, (default=True)

            embeddings_output_dim: int, (default=4)

            embeddings_initializer: str or object, (default='uniform')
                Initializer for the `embeddings` matrix.

            embeddings_regularizer: str or object, (default=None)
                Regularizer function applied to the `embeddings` matrix.

            dense_dropout: float, (default=0) between 0 and 1
                Fraction of the dense input units to drop.

            embedding_dropout: float, (default=0.3) between 0 and 1
                Fraction of the embedding input units to drop.

            stacking_op: str, (default='add')
                - add
                - concat

            output_use_bias: bool, (default=True)

            apply_class_weight: bool, (default=False)

            optimizer: str or object, (default='auto')
                - auto
                - str
                - object

            loss: str or object, (default='auto')

            dnn_params: dict, (default={'hidden_units': ((128, 0, False), (64, 0, False)),
                                        'dnn_activation': 'relu'})

            autoint_params:dict, (default={'num_attention': 3,'num_heads': 1,
                                            'dropout_rate': 0,'use_residual': True})

                fgcnn_params={'fg_filters': (14, 16),
                              'fg_widths': (7, 7),
                              'fg_pool_widths': (2, 2),
                              'fg_new_feat_filters': (2, 2),
                              },
                fibinet_params={
                    'senet_pooling_op': 'mean',
                    'senet_reduction_ratio': 3,
                    'bilinear_type': 'field_interaction',
                },
                cross_params={
                    'num_cross_layer': 4,
                },
                pnn_params={
                    'outer_product_kernel_type': 'mat',
                },
                afm_params={
                    'attention_factor': 4,
                    'dropout_rate': 0
                },
                cin_params={
                    'cross_layer_size': (128, 128),
                    'activation': 'relu',
                    'use_residual': False,
                    'use_bias': False,
                    'direct': False,
                    'reduce_D': False,
                },

            home_dir: str, (default=None)
                The home directory for saving model-related files. Each time running `fit(...)`
                or `fit_cross_validation(...)`, a subdirectory with a time-stamp will be created
                in this directory.

            monitor_metric: str, (default=None)

            earlystopping_patience: int, (default=1)

            gpu_usage_strategy: str, (default='memory_growth')
                - memory_growth
                - None

            distribute_strategy: tensorflow.python.distribute.distribute_lib.Strategy, (default=None)
                -

    Attributes
    ----------
        task: str
            Type of prediction problem, if 'config.task = None'(by default), it will be inferred
            base on the values of `y` when calling 'fit(...)' or 'fit_cross_validation(...)'.
            -'binary' : binary classification task
            -'multiclass' multiclass classfication task
            -'regression' regression task

        num_classes: int
            The number of classes, used only when task is multiclass.

        pos_label: str or int
            The label of positive class, used only when task is binary.

        output_path: str
            Path to directory used to save models. In addition, if a valid 'X_test' is passed into
            `fit_cross_validation(...)`, the prediction results of the test set will be saved in
            this path as well.
            The path is a subdirectory with time-stamp created in the `home directory`. `home directory`
            is specified through `config.home_dir`, if `config.home_dir=None` `output_path` will be created
            in working directory.

        preprocessor: AbstractPreprocessor (default = DefaultPreprocessor)
            Preprocessor is used to perform datasets preprocessing, such as categorization, label encoding,
            imputation, discretization, etc., before feeding into neural nets.

        nets: list(str)
            List of the network cells used to build the DeepModel

        monitor: str
            The metric for monitoring the quality of model in early_stopping, if not specified, the
            first metric in [config.metrics] will be used. (e.g. log_loss/auc_val/accuracy_val...)

        modelset: ModelSet
            The models produced by `fit(...)` or `fit_cross_validation(...)`

        best_model: Model
            A set of models will be produced by `fit_cross_validation(...)`, instead of only one
            model by `fit(...)`. The Best Model is the model with best performance on specific metric.
            The first metric in [config.metrics] will be used by default.

        leaderboard: pandas.DataFrame
            List sorted by specific metric with some meta information and scores. The first metric
            in [config.metrics] will be used by default.

    References
    ----------
    .. [1] ``_

    See also
    --------

    Examples
    --------
    >>>X_train = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
    >>>X_eval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
    >>>y_train = X_train.pop('survived')
    >>>y_eval = X_eval.pop('survived')
    >>>
    >>>config = ModelConfig(nets=deepnets.DeepFM, fixed_embedding_dim=True, embeddings_output_dim=4, auto_discrete=True)
    >>>dt = DeepTable(config=config)
    >>>
    >>>model, history = dt.fit(train, y_train, epochs=100)
    >>>preds = dt.predict(X_eval)
    """

    def __init__(self, config=None, preprocessor=None):
        if config is None:
            config = ModelConfig()
        self.config = config
        self.nets = config.nets
        self.output_path = self._prepare_output_dir(config.home_dir, self.nets)

        self.preprocessor = preprocessor
        self.__current_model = None
        self.__modelset = modelset.ModelSet(metric=self.config.first_metric_name,
                                            best_mode=consts.MODEL_SELECT_MODE_AUTO)

    @property
    def task(self):
        return self.preprocessor.task

    @property
    def num_classes(self):
        return len(self.preprocessor.labels)

    @property
    def classes_(self):
        return self.preprocessor.labels

    @property
    def pos_label(self):
        if self.config.pos_label is not None:
            return self.config.pos_label
        else:
            return self.preprocessor.pos_label

    @property
    def monitor(self):
        monitor = self.config.monitor_metric
        if monitor is None:
            if self.config.metrics is not None and len(self.config.metrics) > 0:
                monitor = 'val_' + self.config.first_metric_name
        return monitor

    @property
    def modelset(self):
        return self.__modelset

    @property
    def best_model(self):
        return self.__modelset.best_model().model

    @property
    def leaderboard(self):
        return self.__modelset.leaderboard()

    def fit(self, X=None, y=None, batch_size=128, epochs=1, verbose=1, callbacks=None,
            validation_split=0.2, validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False):
        logger.info(f'X.Shape={np.shape(X)}, y.Shape={np.shape(y)}, batch_size={batch_size}, config={self.config}')
        logger.info(f'metrics:{self.config.metrics}')
        if np.ndim(X) != 2:
            raise ValueError("Input train data should be 2d .")
        X_shape = np.shape(X)
        if X_shape[1] < 1:
            raise ValueError("Input train data should has 1 feature at least.")
        self.__modelset.clear()

        if self.preprocessor is None:
            self.preprocessor = _get_default_preprocessor(self.config, X, y)

        X, y = self.preprocessor.fit_transform(X, y)
        if validation_data is not None:
            validation_data = self.preprocessor.transform(*validation_data)

        logger.info(f'Training...')
        if class_weight is None and self.config.apply_class_weight and self.task != consts.TASK_REGRESSION:
            class_weight = self.get_class_weight(y)

        callbacks = self.__inject_callbacks(callbacks)
        model = DeepModel(self.task, self.num_classes, self.config,
                          self.preprocessor.categorical_columns,
                          self.preprocessor.continuous_columns,
                          var_categorical_len_columns=self.preprocessor.var_len_categorical_columns)
        history = model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=shuffle,
                            validation_split=validation_split, validation_data=validation_data,
                            validation_steps=validation_steps, validation_freq=validation_freq,
                            callbacks=callbacks, class_weight=class_weight, sample_weight=sample_weight,
                            initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
                            max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
        name = f'{"+".join(self.nets)}'
        logger.info(f'Training finished.')
        self.__set_model('val', name, model, history.history)
        return model, history

    def fit_cross_validation(self, X, y, X_eval=None, X_test=None, num_folds=5, stratified=False, iterators=None,
                             batch_size=None, epochs=1, verbose=1, callbacks=None, n_jobs=1, random_state=9527,
                             shuffle=True, class_weight=None, sample_weight=None,
                             initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1,
                             max_queue_size=10, workers=1, use_multiprocessing=False,
                             oof_metrics=None,
                             ):
        start = time.time()
        logger.info("Start cross validation")
        logger.info(f'X.Shape={np.shape(X)}, y.Shape={np.shape(y)}, batch_size={batch_size}, config={self.config}')
        logger.info(f'metrics:{self.config.metrics}')
        self.__modelset.clear()

        if self.preprocessor is None:
            self.preprocessor = _get_default_preprocessor(self.config, X, y)

        X, y = self.preprocessor.fit_transform(X, y)

        if X_eval is not None:
            logger.info(f'transform X_eval')
            X_eval = self.preprocessor.transform_X(X_eval)
        if X_test is not None:
            logger.info(f'transform X_test')
            X_test = self.preprocessor.transform_X(X_test)

        if iterators is None:
            if is_dask_installed and DaskToolBox.exist_dask_object(X, y):
                iterators = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
            elif stratified and self.task != consts.TASK_REGRESSION:
                iterators = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
            else:
                iterators = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        logger.info(f'Iterators:{iterators}')

        tb = get_tool_box(X)
        X_shape = tb.get_shape(X)
        test_proba_mean = None
        eval_proba_mean = None
        if self.task in (consts.TASK_MULTICLASS, consts.TASK_MULTILABEL):
            oof_proba = np.full((X_shape[0], self.num_classes), np.nan)
        else:
            oof_proba = np.full((X_shape[0], 1), np.nan)

        if is_dask_installed and DaskToolBox.exist_dask_object(X, y):
            X = DaskToolBox.reset_index(DaskToolBox.to_dask_frame_or_series(X))
            y = DaskToolBox.to_dask_type(y)
            if DaskToolBox.is_dask_dataframe_or_series(y):
                y = y.to_dask_array(lengths=True)
            X, y = dask.persist(X, y)
            to_split = np.arange(X_shape[0]), None
        else:
            y = np.array(y)
            to_split = X, y

        if class_weight is None and self.config.apply_class_weight and self.task == consts.TASK_BINARY:
            class_weight = self.get_class_weight(y)

        callbacks = self.__inject_callbacks(callbacks)

        parallel = Parallel(n_jobs=n_jobs, verbose=verbose)

        fit_and_score_kwargs = dict(
            batch_size=batch_size, epochs=epochs, verbose=verbose,
            callbacks=callbacks, class_weight=class_weight, shuffle=shuffle, sample_weight=sample_weight,
            validation_steps=validation_steps, validation_freq=validation_freq,
            initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
            max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing
        )
        oof_scores = [] if oof_metrics is not None else None

        with parallel:
            out = parallel(delayed(_fit_and_score)(
                self.task, self.num_classes, self.config,
                self.preprocessor.categorical_columns, self.preprocessor.continuous_columns,
                n_fold, valid_idx,
                tb.select_df(X, train_idx), y[train_idx], tb.select_df(X, valid_idx), y[valid_idx],
                X_eval, X_test, f'{self.output_path}{"_".join(self.nets)}-kfold-{n_fold + 1}.h5',
                **fit_and_score_kwargs)
                           for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(*to_split)))

            for n_fold, idx, history, fold_oof_proba, fold_eval_proba, fold_test_proba in out:
                oof_proba[idx] = fold_oof_proba
                if X_eval is not None:
                    if eval_proba_mean is None:
                        eval_proba_mean = fold_eval_proba / num_folds
                    else:
                        eval_proba_mean += fold_eval_proba / num_folds
                if X_test is not None:
                    if test_proba_mean is None:
                        test_proba_mean = fold_test_proba / num_folds
                    else:
                        test_proba_mean += fold_test_proba / num_folds
                if oof_metrics is not None:
                    fold_y_true = y[idx]
                    if self.task == consts.TASK_BINARY:
                        fold_y_proba = self._fix_softmax_proba(fold_oof_proba.shape[0], fold_oof_proba.copy())
                    else:
                        fold_y_proba = fold_oof_proba.copy()
                    fold_y_true = self.preprocessor.inverse_transform_y(fold_y_true)
                    fold_y_pred = self.proba2predict(fold_y_proba, encode_to_label=True)
                    oof_scores.append(tb.metrics.calc_score(fold_y_true, fold_y_pred, fold_y_proba, task=self.task,
                                                            metrics=oof_metrics, pos_label=self.pos_label,
                                                            classes=self.classes_))

                self.__push_model('val', f'{"+".join(self.nets)}-kfold-{n_fold + 1}',
                                  f'{self.output_path}{"_".join(self.nets)}-kfold-{n_fold + 1}.h5', history)
        oof_proba_origin = oof_proba.copy()
        nan_idx = np.argwhere(np.isnan(oof_proba).any(1)).ravel()
        if self.task == consts.TASK_BINARY:
            oof_proba_fixed = self._fix_softmax_proba(X.shape[0], oof_proba_origin.copy())
        elif self.task == consts.TASK_REGRESSION:
            oof_proba_fixed = oof_proba_origin.reshape(X.shape[0])
        else:
            oof_proba_fixed = oof_proba_origin

        if len(nan_idx) > 0:
            oof_proba_fixed[nan_idx] = np.nan

        if eval_proba_mean is not None:
            if self.task == consts.TASK_BINARY:
                eval_proba_mean_fixed = self._fix_softmax_proba(X_eval.shape[0], eval_proba_mean.copy())
            else:
                eval_proba_mean_fixed = eval_proba_mean
        else:
            eval_proba_mean_fixed = eval_proba_mean

        if test_proba_mean is not None:
            if self.task == consts.TASK_BINARY:
                test_proba_mean_fixed = self._fix_softmax_proba(X_test.shape[0], test_proba_mean.copy())
                file = f'{self.output_path}{"_".join(self.nets)}-cv-{num_folds}.csv'
                with fs.open(file, 'w', encoding='utf-8') as f:
                    pd.DataFrame(test_proba_mean.reshape(-1)).to_csv(f, index=False)

            else:
                test_proba_mean_fixed = test_proba_mean
        else:
            test_proba_mean_fixed = test_proba_mean

        logger.info(f'fit_cross_validation taken {time.time() - start}s')

        if oof_metrics is not None:
            return oof_proba_fixed, eval_proba_mean_fixed, test_proba_mean_fixed, oof_scores
        else:
            return oof_proba_fixed, eval_proba_mean_fixed, test_proba_mean_fixed

    def _fix_softmax_proba(self, n_rows, proba):
        # proba shape should be (n, 1) if output layer is softmax
        if proba is None:
            return None
        else:
            # assert proba.shape == (n_rows, 1)
            # return np.insert(proba, 0, values=(1 - proba).reshape(1, -1), axis=1)
            return get_tool_box(proba).fix_binary_predict_proba_result(proba)

    def evaluate(self, X_test, y_test, batch_size=256, verbose=0,
                 model_selector=consts.MODEL_SELECTOR_CURRENT, return_dict=True):
        X_t, y_t = self.preprocessor.transform(X_test, y_test)
        # y_t = np.array(y_t)
        model = self.get_model(model_selector)
        if not isinstance(model, DeepModel):
            raise ValueError(f'Wrong model_selector:{model_selector}')
        result = model.evaluate(X_t, y_t, batch_size=batch_size, verbose=verbose, return_dict=return_dict)
        return result

    def predict_proba(self, X, batch_size=128, verbose=0,
                      model_selector=consts.MODEL_SELECTOR_CURRENT, auto_transform_data=True, ):
        n_rows = X.shape[0]
        start = time.time()
        if model_selector == consts.MODEL_SELECTOR_ALL:
            models = self.get_model(model_selector)
            proba_avg = None
            if auto_transform_data:
                X = self.preprocessor.transform_X(X)
            for model in models:
                proba = self.__predict(model, X, batch_size=batch_size, verbose=verbose, auto_transform_data=False)
                if proba_avg is None:
                    proba_avg = np.zeros(proba.shape)
                proba_avg += proba
            proba_avg /= len(models)
            proba = proba_avg
        else:
            proba = self.__predict(self.get_model(model_selector),
                                   X, batch_size=batch_size,
                                   verbose=verbose,
                                   auto_transform_data=auto_transform_data)
        logger.info(f'predict_proba taken {time.time() - start}s')
        return proba

    def predict_proba_all(self, X, batch_size=128, verbose=0, auto_transform_data=True, ):
        mis = self.__modelset.get_modelinfos()
        proba_all = {}
        if auto_transform_data:
            X = self.preprocessor.transform_X(X)
        for mi in mis:
            model = self.get_model(mi.name)
            proba = self.__predict(model, X, batch_size=batch_size, verbose=verbose, auto_transform_data=False)
            proba_all[mi.name] = proba
        return proba_all

    def predict(self, X, encode_to_label=True, batch_size=128, verbose=0,
                model_selector=consts.MODEL_SELECTOR_CURRENT, auto_transform_data=True):
        proba = self.predict_proba(X, batch_size, verbose,
                                   model_selector=model_selector,
                                   auto_transform_data=auto_transform_data)

        return self.proba2predict(proba, encode_to_label)

    def proba2predict(self, proba, encode_to_label=True):
        if self.task == consts.TASK_REGRESSION:
            return proba
        if proba is None:
            raise ValueError('[proba] can not be none.')
        if len(proba.shape) == 1:
            proba = proba.reshape((-1, 1))

        if proba.shape[-1] > 1:
            predict = proba.argmax(axis=-1)
        else:
            predict = (proba > 0.5).astype(consts.DATATYPE_PREDICT_CLASS)
        if encode_to_label:
            logger.info('Reverse indicators to labels.')
            predict = self.preprocessor.inverse_transform_y(predict)

        return predict

    def apply(self, X, output_layers, concat_outputs=False, batch_size=128, verbose=0,
              model_selector=consts.MODEL_SELECTOR_CURRENT, auto_transform_data=True, transformer=None):
        start = time.time()

        model = self.get_model(model_selector)
        if not isinstance(model, DeepModel):
            raise ValueError(f'Wrong model_selector:{model_selector}')
        if auto_transform_data:
            X = self.preprocessor.transform_X(X)
        output = model.apply(X, output_layers, concat_outputs, batch_size, verbose, transformer)
        logger.info(f'apply taken {time.time() - start}s')
        return output

    def concat_emb_dense(self, flatten_emb_layer, dense_layer):
        x = None
        if flatten_emb_layer is not None and dense_layer is not None:
            x = Concatenate(name='concat_embedding_dense')([flatten_emb_layer, dense_layer])
        elif flatten_emb_layer is not None:
            x = flatten_emb_layer
        elif dense_layer is not None:
            x = dense_layer
        else:
            raise ValueError('No input layer exists.')
        x = BatchNormalization(name='bn_concat_emb_dense')(x)
        logger.info(f'Concat embedding and dense layer shape:{x.shape}')
        return x

    def get_model(self, model_selector=consts.MODEL_SELECTOR_CURRENT, ):
        if model_selector == consts.MODEL_SELECTOR_CURRENT:
            # get model by name
            mi = self.__modelset.get_modelinfo(self.__current_model)
        elif model_selector == consts.MODEL_SELECTOR_BEST:
            mi = self.__modelset.best_model()
        elif model_selector == consts.MODEL_SELECTOR_ALL:
            ms = []
            for mi in self.__modelset.get_modelinfos():
                if isinstance(mi.model, str):
                    dm = self.load_deepmodel(mi.model)
                    mi.model = dm
                ms.append(mi.model)
            return ms
        else:
            # get model by name
            mi = self.__modelset.get_modelinfo(model_selector)
        if mi is None:
            raise ValueError(f'{model_selector} does not exsit.')

        if isinstance(mi.model, str):
            dm = self.load_deepmodel(mi.model)
            mi.model = dm
        return mi.model

    def get_class_weight(self, y):
        # logger.info('Calc classes weight.')
        # if len(y.shape) == 1:
        #     y = to_categorical(y)
        # y_sum = y.sum(axis=0)
        # class_weight = {}
        # total = y.shape[0]
        # classes = len(y_sum)
        # logger.info(f"Examples:\nTotal:{total}")
        # for i in range(classes):
        #     weight = total / y_sum[i] / classes
        #     class_weight[i] = weight
        #     logger.info(f'class {i}:{weight}')

        n = len(self.classes_)
        class_weight = get_tool_box(y).compute_class_weight('balanced', classes=range(n), y=y)
        class_weight = {k: v for k, v in zip(range(n), class_weight)}
        logger.info(f'classes weight: {class_weight}')

        return class_weight

    def _prepare_output_dir(self, home_dir, nets):
        if home_dir is None:
            home_dir = 'dt_output'
        if home_dir[-1] == '/':
            home_dir = home_dir[:-1]

        running_dir = f'dt_{time.strftime("%Y%m%d%H%M%S")}_{"_".join(nets)}'
        output_path = os.path.expanduser(f'{home_dir}/{running_dir}/')
        if not fs.exists(output_path):
            fs.makedirs(output_path, exist_ok=True)
        return output_path

    def __predict(self, model, X, batch_size=128, verbose=0, auto_transform_data=True, ):
        logger.info("Perform prediction...")
        if auto_transform_data:
            X = self.preprocessor.transform_X(X)
        proba = model.predict(X, batch_size=batch_size, verbose=verbose)
        if self.task == consts.TASK_BINARY:
            # return self._fix_softmax_proba(X.shape[0], proba)
            return get_tool_box(proba).fix_binary_predict_proba_result(proba)
        else:
            return proba

    def __set_model(self, type, name, model, history):
        self.__modelset.clear()
        self.__push_model(type, name, model, history)

    def __push_model(self, type, name, model, history, save_model=True):
        modelfile = ''
        if save_model and isinstance(model, DeepModel):
            modelfile = f'{self.output_path}{name}.h5'
            model.save(modelfile)
            logger.info(f'Model has been saved to:{modelfile}')
        mi = modelset.ModelInfo(type, name, model, {}, history=history, modelfile=modelfile)
        self.__modelset.push(mi)
        self.__current_model = mi.name

    def __inject_callbacks(self, callbacks):
        # mcp = None
        es = None
        if callbacks is not None:
            for callback in callbacks:
                # if isinstance(callback, ModelCheckpoint):
                #   mcp = callback
                if isinstance(callback, EarlyStopping):
                    es = callback
        else:
            callbacks = []

        if self.config.earlystopping_mode == 'auto':
            if self.monitor.lower() in ['auc', 'acc', 'accuracy', 'precision', 'recall', 'f1',
                                        'val_auc', 'val_acc', 'val_accuracy', 'val_precision', 'val_recall', 'val_f1']:
                mode = 'max'
            else:
                mode = 'min'
        else:
            mode = self.config.earlystopping_mode
        # if mcp is None:
        #    mcp = ModelCheckpoint(self.model_filepath,
        #                          monitor=self.monitor,
        #                          verbose=0,
        #                          save_best_only=False,
        #                          save_weights_only=False,
        #                          mode=mode,
        #                          save_freq='epoch',
        #                          )
        #    callbacks.append(mcp)
        #    logger.info(f'Injected a callback [ModelCheckpoint].\nfilepath:{mcp.filepath}\nmonitor:{mcp.monitor}')
        es_patience = self.config.earlystopping_patience
        if es is None and isinstance(es_patience, int) and es_patience > 0:
            es = EarlyStopping(monitor=self.monitor if tf_less_than('2.2') else self.monitor.lower(),
                               restore_best_weights=True,
                               patience=es_patience,
                               verbose=1,
                               # min_delta=0.0001,
                               mode=mode,
                               baseline=None,
                               )
            callbacks.append(es)
            if logger.is_info_enabled():
                logger.info(f'Injected a callback [EarlyStopping]. monitor:{es.monitor}, '
                            f'patience:{es.patience}, mode:{mode}')
        return callbacks

    def __getstate__(self):
        import copy

        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__.copy()

        if self.config.distribute_strategy is not None:
            tmp_conf = self.config._replace(distribute_strategy=None)
            tmp_preprocessor = copy.deepcopy(self.preprocessor)
            tmp_preprocessor.config = self.preprocessor.config._replace(distribute_strategy=None)
            state['config'] = tmp_conf
            state['preprocessor'] = tmp_preprocessor

        return state

    def save(self, filepath, deepmodel_basename=None):
        if filepath[-1] != '/':
            filepath = filepath + '/'

        if not fs.exists(filepath):
            fs.makedirs(filepath, exist_ok=True)
        num_model = len(self.__modelset.get_modelinfos())
        for mi in self.__modelset.get_modelinfos():
            if isinstance(mi.model, str):
                dm = self.load_deepmodel(mi.model)
                mi.model = dm
            if not isinstance(mi.model, DeepModel):
                raise ValueError(f'Currently does not support saving non-DeepModel models.')

            if num_model == 1 and deepmodel_basename is not None:
                mi.name = deepmodel_basename
                self.__current_model = deepmodel_basename
            modelfile = f'{filepath}{mi.name}.h5'
            mi.model.save(modelfile)
            mi.model = modelfile

        with fs.open(f'{filepath}dt.pkl', 'wb') as output:
            pickle.dump(self, output, protocol=4)

    @staticmethod
    def load(filepath):
        if filepath[-1] != '/':
            filepath = filepath + '/'
        with fs.open(f'{filepath}dt.pkl', 'rb') as input:
            dt = pickle.load(input)
            dt.restore_modelset(filepath)
            return dt

    def restore_modelset(self, filepath):
        for mi in self.__modelset.get_modelinfos():
            if isinstance(mi.model, str):
                modelfile = mi.model
                modelfile = os.path.split(modelfile)[-1]
                dm = self.load_deepmodel(f'{filepath}{modelfile}')
                mi.model = dm

    def load_deepmodel(self, filepath):
        if fs.exists(filepath):
            logger.info(f'Load model from: {filepath}.')
            dm = DeepModel(self.task, self.num_classes, self.config,
                           self.preprocessor.categorical_columns,
                           self.preprocessor.continuous_columns, model_file=filepath)
            return dm
        else:
            raise ValueError(f'Invalid model filename:{filepath}.')


def _fit_and_score(task, num_classes, config, categorical_columns, continuous_columns,
                   n_fold, valid_idx, X_train, y_train, X_val, y_val,
                   X_eval=None, X_test=None, model_file=None,
                   batch_size=128, epochs=1, verbose=0, callbacks=None,
                   shuffle=True, class_weight=None, sample_weight=None,
                   initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1,
                   max_queue_size=10, workers=1, use_multiprocessing=False):
    logger.info(f'\nFold:{n_fold + 1}\n')
    model = DeepModel(task, num_classes, config, categorical_columns, continuous_columns)
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                        callbacks=callbacks, validation_data=(X_val, y_val),
                        shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight,
                        initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps, validation_freq=validation_freq,
                        max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
    logger.info(f'Fold {n_fold + 1} fitting over.')

    oof_proba = model.predict(X_val)
    eval_proba = model.predict(X_eval) if X_eval is not None else None
    test_proba = model.predict(X_test) if X_test is not None else None
    oof_proba, eval_proba, test_proba = get_tool_box(X_val).to_local(oof_proba, eval_proba, test_proba)
    logger.info(f'Fold {n_fold + 1} scoring over.')

    if model_file is not None:
        model.save(model_file)
        logger.info(f'Save model to:{model_file}.')

        if X_test is not None:
            file = f'{model_file}.test_proba.csv'
            with fs.open(file, 'w', encoding='utf-8') as f:
                pd.DataFrame(test_proba).to_csv(f, index=False)

    model.release()
    return n_fold, valid_idx, history.history, oof_proba, eval_proba, test_proba


def probe_evaluate(dt, X, y, X_test, y_test, layers, score_fn={}):
    from sklearn.linear_model import LogisticRegression
    logger.info('Extracting features of train set...')
    features_train = dt.apply(X, output_layers=layers)
    logger.info('Extracting features of test set...')
    features_test = dt.apply(X_test, output_layers=layers)
    y = dt.preprocessor.transform_y(y)
    y_test = dt.preprocessor.transform_y(y_test)

    if not isinstance(features_train, list):
        features_train = [features_train]
        features_test = [features_test]

    result = {}
    for i, x_train in enumerate(features_train):
        clf = LogisticRegression(random_state=0).fit(x_train, y)
        logger.info(f'Fit model for layer[{layers[i]}]...')
        y_proba = clf.predict_proba(features_test[i])[:, 1]
        y_score = clf.predict(features_test[i])
        logger.info(f'Scoring...')
        if len(score_fn) == 0:
            score = clf.score(features_test[i], y_test)
            logger.info(f'Evaluating accuracy score...')
            result[layers[i]] = {'accuracy': score}
        else:
            result[layers[i]] = {}
            for metric in score_fn.keys():
                logger.info(f'Evaluating {metric} score...')
                fn = score_fn[metric]
                if fn == roc_auc_score:
                    score = fn(y_test, y_proba)
                else:
                    score = fn(y_test, y_score)
                result[layers[i]][metric] = score
                logger.info(f'{metric}:{score}')
            # result[layers[i]] = {metric:score_fn[metric](features_test[i], y_score) for metric in score_fn.keys()}
    return result


def _get_default_preprocessor(config, X, y):
    if is_dask_installed and DaskToolBox.exist_dask_object(X, y):
        return DefaultDaskPreprocessor(config, compute_to_local=False)
    else:
        return DefaultPreprocessor(config)
