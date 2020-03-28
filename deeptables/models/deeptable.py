# -*- coding:utf-8 -*-
"""Training and inference for tabular datasets using neural nets."""

import datetime
import os

import numpy as np
import time
import pandas as pd
import pickle
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate, BatchNormalization
from tensorflow.keras.utils import to_categorical

from . import modelset, deepnets
from .config import ModelConfig
from .deepmodel import DeepModel
from .preprocessor import DefaultPreprocessor
from ..utils import dt_logging, consts

logger = dt_logging.get_logger()


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
                Typically you will use `metrics=['accuracy']` or `metrics=['AUC']`.
                Every metric should be a built-in evaluation metric in tf.keras.metrics or a callable object
                like `r2(y_true, y_pred):...` .
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

            dnn_params: dict, (default={'dnn_units': ((128, 0, False), (64, 0, False)),
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

    def __init__(self, config=None):
        if config is None:
            config = ModelConfig()
        self.config = config
        self.nets = config.nets
        self.output_path = self._prepare_output_dir(config.home_dir, self.nets)
        self.preprocessor = DefaultPreprocessor(config)
        self.__current_model = None
        self.__modelset = modelset.ModelSet(metric=self.config.first_metric_name, best_mode=consts.MODEL_SELECT_MODE_AUTO)

    @property
    def task(self):
        return self.preprocessor.task

    @property
    def num_classes(self):
        return len(self.preprocessor.labels)

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
        self.__modelset.clear()

        X, y = self.preprocessor.fit_transform(X, y)
        if validation_data is not None:
            if len(validation_data) != 2:
                raise ValueError(f'Unexpected validation_data length, expected 2 but {len(validation_data)}.')
            X_val, y_val = validation_data(0), validation_data(1)
            X_val, y_val = self.preprocessor.transform(X_val, y_val)
            validation_data = (X_val, y_val)

        logger.info(f'training...')
        if class_weight is None and self.config.apply_class_weight and self.task != consts.TASK_REGRESSION:
            class_weight = self.get_class_weight(y)

        callbacks = self.__inject_callbacks(callbacks)
        model = DeepModel(self.task, self.num_classes, self.config,
                          self.preprocessor.categorical_columns,
                          self.preprocessor.continuous_columns)
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
                             max_queue_size=10, workers=1, use_multiprocessing=False
                             ):
        print("Start cross validation")
        start = time.time()
        logger.info(f'X.Shape={np.shape(X)}, y.Shape={np.shape(y)}, batch_size={batch_size}, config={self.config}')
        logger.info(f'metrics:{self.config.metrics}')
        self.__modelset.clear()

        X, y = self.preprocessor.fit_transform(X, y)

        if X_eval is not None:
            print(f'transform X_eval')
            X_eval = self.preprocessor.transform_X(X_eval)
        if X_test is not None:
            print(f'transform X_test')
            X_test = self.preprocessor.transform_X(X_test)

        if iterators is None:
            if stratified and self.task != consts.TASK_REGRESSION:
                iterators = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
            else:
                iterators = KFold(n_splits=num_folds, shuffle=True, random_state=random_state)
        print(f'Iterators:{iterators}')

        test_proba_mean = None
        eval_proba_mean = None
        if self.task == consts.TASK_MULTICLASS:
            oof_proba = np.zeros((y.shape[0], self.num_classes))
        else:
            oof_proba = np.zeros((y.shape[0], 1))

        y = np.array(y)
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
        with parallel:
            out = parallel(delayed(_fit_and_score)(
                self.task, self.num_classes, self.config,
                self.preprocessor.categorical_columns, self.preprocessor.continuous_columns,
                n_fold, valid_idx,
                X.iloc[train_idx], y[train_idx], X.iloc[valid_idx], y[valid_idx],
                X_eval, X_test, f'{self.output_path}{"_".join(self.nets)}-kfold-{n_fold + 1}.h5',
                **fit_and_score_kwargs)
                           for n_fold, (train_idx, valid_idx) in enumerate(iterators.split(X, y)))

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
                self.__push_model('val', f'{"+".join(self.nets)}-kfold-{n_fold + 1}',
                                  f'{self.output_path}{"_".join(self.nets)}-kfold-{n_fold + 1}.h5', history)

        if oof_proba.shape[-1] == 1:
            oof_proba = oof_proba.reshape(-1)
        if eval_proba_mean is not None and eval_proba_mean.shape[-1] == 1:
            eval_proba_mean = eval_proba_mean.reshape(-1)
        if test_proba_mean is not None and test_proba_mean.shape[-1] == 1:
            test_proba_mean = test_proba_mean.reshape(-1)
            file = f'{self.output_path}{"_".join(self.nets)}-cv-{num_folds}.csv'
            pd.DataFrame(test_proba_mean).to_csv(file, index=False)
        print(f'fit_cross_validation cost:{time.time() - start}')
        return oof_proba, eval_proba_mean, test_proba_mean

    def evaluate(self, X_test, y_test, batch_size=256, verbose=0, model_selector=consts.MODEL_SELECTOR_CURRENT, ):
        X_t, y_t = self.preprocessor.transform(X_test, y_test)
        y_t = np.array(y_t)
        model = self.get_model(model_selector)
        if not isinstance(model, DeepModel):
            raise ValueError(f'Wrong model_selector:{model_selector}')
        result = model.evaluate(X_t, y_t, batch_size=batch_size, verbose=verbose)
        return result

    def predict_proba(self, X, batch_size=128, verbose=0,
                      model_selector=consts.MODEL_SELECTOR_CURRENT, auto_transform_data=True, ):
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
            print(f'predict_proba cost:{time.time() - start}')
            return proba_avg
        else:
            proba = self.__predict(self.get_model(model_selector),
                                   X, batch_size=batch_size,
                                   verbose=verbose,
                                   auto_transform_data=auto_transform_data)
            print(f'predict_proba cost:{time.time() - start}')
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
        print(f'apply cost:{time.time() - start}')
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
        print(f'Concat embedding and dense layer shape:{x.shape}')
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
        print('Calc classes weight.')
        if len(y.shape) == 1:
            y = to_categorical(y)
        y_sum = y.sum(axis=0)
        class_weight = {}
        total = y.shape[0]
        classes = len(y_sum)
        print(f"Examples:\nTotal:{total}")
        for i in range(classes):
            weight = total / y_sum[i] / classes
            class_weight[i] = weight
            print(f'class {i}:{weight}')

        return class_weight

    def _prepare_output_dir(self, home_dir, nets):
        if home_dir is None:
            home_dir = 'dt_output'
        if home_dir[-1] == '/':
            home_dir = home_dir[:-1]

        running_dir = f'dt_{datetime.datetime.now().__format__("%Y%m%d %H%M%S")}_{"_".join(nets)}'
        output_path = os.path.expanduser(f'{home_dir}/{running_dir}/')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        return output_path

    def __predict(self, model, X, batch_size=128, verbose=0, auto_transform_data=True, ):
        logger.info("Perform prediction...")
        if auto_transform_data:
            X = self.preprocessor.transform_X(X)
        return model.predict(X, batch_size=batch_size, verbose=verbose)

    def __set_model(self, type, name, model, history):
        self.__modelset.clear()
        self.__push_model(type, name, model, history)

    def __push_model(self, type, name, model, history, save_model=True):
        modelfile = ''
        if save_model and isinstance(model, DeepModel):
            modelfile = f'{self.output_path}{name}.h5'
            model.save(modelfile)
            print(f'Model has been saved to:{modelfile}')
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

        if 'auc' in self.monitor.lower() or 'acc' in self.monitor.lower():
            mode = 'max'
        else:
            mode = 'min'
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
        #    print(f'Injected a callback [ModelCheckpoint].\nfilepath:{mcp.filepath}\nmonitor:{mcp.monitor}')
        if es is None:
            es = EarlyStopping(monitor=self.monitor,
                               restore_best_weights=True,
                               patience=self.config.earlystopping_patience,
                               verbose=1,
                               min_delta=0.001,
                               mode=mode,
                               baseline=None,
                               )
            callbacks.append(es)
            print(f'Injected a callback [EarlyStopping]. monitor:{es.monitor}, patience:{es.patience}, mode:{mode}')
        return callbacks

    def save(self, filepath):
        if filepath[-1] != '/':
            filepath = filepath + '/'

        if not os.path.exists(filepath):
            os.mkdirs(filepath)

        for mi in self.__modelset.get_modelinfos():
            if isinstance(mi.model, str):
                dm = self.load_deepmodel(mi.model)
                mi.model = dm
            if not isinstance(mi.model, DeepModel):
                raise ValueError(f'Currently does not support saving non-DeepModel models.')
            modelfile = f'{filepath}{mi.name}.h5'
            mi.model.save(modelfile)
            mi.model = modelfile

        with open(f'{filepath}dt.pkl', 'wb') as output:
            pickle.dump(self, output, protocol=2)

    @staticmethod
    def load(filepath):
        if filepath[-1] != '/':
            filepath = filepath + '/'
        with open(f'{filepath}dt.pkl', 'rb') as input:
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
        if os.path.exists(filepath):
            print(f'Load model from disk:{filepath}.')
            dm = DeepModel(self.task, self.num_classes, self.config,
                           self.preprocessor.categorical_columns, self.preprocessor.continuous_columns, filepath)
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
    print(f'\nFold:{n_fold + 1}\n')
    model = DeepModel(task, num_classes, config, categorical_columns, continuous_columns)
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                        callbacks=callbacks, validation_data=(X_val, y_val),
                        shuffle=shuffle, class_weight=class_weight, sample_weight=sample_weight,
                        initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps, validation_freq=validation_freq,
                        max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing)
    print(f'Fold {n_fold + 1} fitting over.')
    oof_proba = model.predict(X_val)
    eval_proba = None
    test_proba = None
    if X_eval is not None:
        eval_proba = model.predict(X_eval)
    if X_test is not None:
        test_proba = model.predict(X_test)
        if model_file is not None:
            file = f'{model_file}.test_proba.csv'
            pd.DataFrame(test_proba).to_csv(file, index=False)
    print(f'Fold {n_fold + 1} scoring over.')
    if model_file is not None:
        model.save(model_file)
        print(f'Save model to:{model_file}.')
    model.release()
    return n_fold, valid_idx, history.history, oof_proba, eval_proba, test_proba


def infer_task_type(y):
    uniques = set(y)
    n_unique = len(uniques)
    labels = []

    if n_unique == 2:
        print(f'2 class detected, {uniques}, so inferred as a [binary classification] task')
        task = consts.TASK_BINARY  # TASK_BINARY
        labels = sorted(uniques)
    else:
        if y.dtype == 'float':
            print(f'Target column type is float, so inferred as a [regression] task.')
            task = consts.TASK_REGRESSION
        else:
            if n_unique > 1000:
                if 'int' in y.dtype:
                    print(
                        'The number of classes exceeds 1000 and column type is int, so inferred as a [regression] task ')
                    task = consts.TASK_REGRESSION
                else:
                    raise ValueError(
                        'The number of classes exceeds 1000, please confirm whether your predict target is correct ')
            else:
                print(f'{n_unique} class detected, inferred as a [multiclass classification] task')
                task = consts.TASK_MULTICLASS
                labels = sorted(uniques)
    return task, labels


def probe_evaluate(dt, X, y, X_test, y_test, layers, score_fn={}):
    from sklearn.linear_model import LogisticRegression
    print('Extracting features of train set...')
    features_train = dt.apply(X, output_layers=layers)
    print('Extracting features of test set...')
    features_test = dt.apply(X_test, output_layers=layers)
    y = dt.preprocessor.transform_y(y)
    y_test = dt.preprocessor.transform_y(y_test)

    if not isinstance(features_train, list):
        features_train = [features_train]
        features_test = [features_test]

    result = {}
    for i, x_train in enumerate(features_train):
        clf = LogisticRegression(random_state=0).fit(x_train, y)
        print(f'Fit model for layer[{layers[i]}]...')
        y_proba = clf.predict_proba(features_test[i])[:, 1]
        y_score = clf.predict(features_test[i])
        print(f'Scoring...')
        if len(score_fn) == 0:
            score = clf.score(features_test[i], y_test)
            print(f'Evaluating accuracy score...')
            result[layers[i]] = {'accuracy': score}
        else:
            result[layers[i]] = {}
            for metric in score_fn.keys():
                print(f'Evaluating {metric} score...')
                fn = score_fn[metric]
                if fn == roc_auc_score:
                    score = fn(y_test, y_proba)
                else:
                    score = fn(y_test, y_score)
                result[layers[i]][metric] = score
                print(f'{metric}:{score}')
            # result[layers[i]] = {metric:score_fn[metric](features_test[i], y_score) for metric in score_fn.keys()}
    return result
