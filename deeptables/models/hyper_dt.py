# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import copy

import pandas as pd
import pickle

from deeptables.models.config import ModelConfig
from deeptables.models.deeptable import DeepTable
from deeptables.models.preprocessor import DefaultPreprocessor
from deeptables.utils import dt_logging, fs, consts as DT_consts
from hypernets.core.search_space import HyperSpace, ModuleSpace, Choice, Bool, MultipleChoice
from hypernets.experiment import make_experiment as _make_experiment
from hypernets.model import Estimator, HyperModel
from hypernets.utils import DocLens, isnotebook

logger = dt_logging.get_logger(__name__)


def _to_hp(v):
    if isinstance(v, (list, tuple)):
        v = Choice(v)
    return v


class DTModuleSpace(ModuleSpace):
    def __init__(self, space=None, name=None, **hyperparams):
        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.space.DT_Module = self
        self.config = None

    def _compile(self):
        self.config = ModelConfig(**self.param_values)

    def _forward(self, inputs):
        return inputs

    def _on_params_ready(self):
        self._compile()


class DTFit(ModuleSpace):
    def __init__(self, space=None, name=None, **hyperparams):
        # if batch_size is None:
        #     batch_size = Choice([128, 256])
        # hyperparams['batch_size'] = batch_size
        #
        # if epochs is not None:
        #     hyperparams['epochs'] = epochs

        for k, v in hyperparams.items():
            hyperparams[k] = _to_hp(v)

        ModuleSpace.__init__(self, space, name, **hyperparams)
        self.space.fit_params = self

    def _compile(self):
        pass

    def _forward(self, inputs):
        return inputs

    def _on_params_ready(self):
        self._compile()


class DnnModule(ModuleSpace):
    def __init__(self, hidden_units=None, reduce_factor=None, dnn_dropout=None, use_bn=None, dnn_layers=None,
                 activation=None, space=None, name=None, **hyperparams):
        if hidden_units is None:
            hidden_units = [100, 200, 300, 500, 800, 1000]
        hyperparams['hidden_units'] = _to_hp(hidden_units)

        if reduce_factor is None:
            reduce_factor = [1, 0.8, 0.5]
        hyperparams['reduce_factor'] = _to_hp(reduce_factor)

        if dnn_dropout is None:
            dnn_dropout = [0, 0.1, 0.3, 0.5]
        hyperparams['dnn_dropout'] = _to_hp(dnn_dropout)

        if use_bn is None:
            use_bn = Bool()
        hyperparams['use_bn'] = use_bn

        if dnn_layers is None:
            dnn_layers = [1, 2, 3]
        hyperparams['dnn_layers'] = _to_hp(dnn_layers)

        if activation is None:
            activation = 'relu'
        hyperparams['activation'] = activation

        ModuleSpace.__init__(self, space, name, **hyperparams)

    def _compile(self):
        dnn_layers = self.param_values['dnn_layers']
        hidden_units = []
        for i in range(0, dnn_layers):
            hidden_units.append(
                (int(self.param_values['hidden_units'] * 1 if i == 0 else (
                        self.param_values['hidden_units'] * (self.param_values['reduce_factor'] ** i))),
                 self.param_values['dnn_dropout'],
                 self.param_values['use_bn']))
        dnn_params = {
            'hidden_units': hidden_units,
            'dnn_activation': self.param_values['activation'],
        }
        self.space.DT_Module.config = self.space.DT_Module.config._replace(dnn_params=dnn_params)

    def _forward(self, inputs):
        return inputs

    def _on_params_ready(self):
        self._compile()


class DTEstimator(Estimator):
    def __init__(self, space_sample, cache_preprocessed_data=False, **config_kwargs):
        Estimator.__init__(self, space_sample=space_sample)

        self.config_kwargs = config_kwargs
        self.cache_preprocessed_data = cache_preprocessed_data
        self.model = self._build_model(space_sample)

        # fitted
        self.classes_ = None

    def _build_model(self, space_sample):
        config = space_sample.DT_Module.config._replace(**self.config_kwargs)
        if self.cache_preprocessed_data:
            preprocessor = DefaultPreprocessor(config)
        else:
            preprocessor = None
        model = DeepTable(config, preprocessor=preprocessor)
        return model

    def summary(self):
        if logger.is_info_enabled():
            try:
                mi = self.model.get_model()
                if mi is not None:
                    mi.model.summary()
            except(Exception) as ex:
                pass
                # logger.info('---------no summary-------------')
                # logger.info(ex)

    def fit(self, X, y, eval_set=None, pos_label=None, n_jobs=1, **kwargs):
        # fit_params = self.space_sample.__dict__.get('fit_params')
        # if fit_params is not None:
        #     kwargs.update(fit_params.param_values)
        if kwargs.get('cross_validation') is not None:
            kwargs.pop('cross_validation')
            self.model.fit_cross_validation(X, y, n_jobs=n_jobs, **kwargs)
        else:
            fit_kwargs = self.space_sample.fit_params.param_values.copy()
            fit_kwargs.update(kwargs)
            self.model.fit(X, y, **fit_kwargs)

        self.classes_ = getattr(self.model, 'classes_', None)
        return self

    def fit_cross_validation(self, X, y, eval_set=None, metrics=None, pos_label=None, **kwargs):
        assert isinstance(metrics, (list, tuple))
        fit_kwargs = self.space_sample.fit_params.param_values.copy()
        fit_kwargs.update(kwargs)
        oof_proba, _, _, oof_scores = self.model.fit_cross_validation(X, y, oof_metrics=metrics, **fit_kwargs)

        # calc final score with mean
        scores = pd.concat([pd.Series(s) for s in oof_scores], axis=1).mean(axis=1).to_dict()
        logger.info(f'fit_cross_validation score:{scores}, folds score:{oof_scores}')

        self.classes_ = getattr(self.model, 'classes_', None)

        return scores, oof_proba, oof_scores

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def evaluate(self, X, y, eval_set=None, metrics=None, **kwargs):
        # scores = self.model.evaluate(X, y, batch_size=256, return_dict=False)
        scores = self.model.evaluate(X, y, batch_size=256, return_dict=False)
        dt_model = self.model.get_model()

        tf_metrics_names = dt_model.model.metrics_names

        user_metrics = dt_model.config.metrics
        if len(scores) != (len(user_metrics) + 1):
            raise ValueError(f"Evaluate result has {len(scores)} items with loss score," +
                             f" not match with user specified metrics {user_metrics}; tf metrics names {tf_metrics_names}")

        loss_name = tf_metrics_names[0]
        ret_metrics = [loss_name]
        ret_metrics.extend(dt_model.config.metrics)

        logger.info(f"TF metrics names is {tf_metrics_names} and user's is {user_metrics}")

        result = dict(zip(ret_metrics, scores))

        return result

    def predict_proba(self, X, **kwargs):
        result = self.model.predict_proba(X, **kwargs)
        return result

    def save(self, model_path):
        if not model_path.endswith(fs.sep):
            model_path = model_path + fs.sep

        self.model.save(model_path)

        stub = copy.copy(self)
        stub.model = None
        stub_path = model_path + 'dt_estimator.pkl'
        with fs.open(stub_path, 'wb') as f:
            pickle.dump(stub, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(model_path):
        if not fs.exists(model_path):
            raise ValueError(f'Not found storage path: {model_path}')

        if not model_path.endswith(fs.sep):
            model_path = model_path + fs.sep

        stub_path = model_path + 'dt_estimator.pkl'
        if not fs.exists(stub_path):
            raise ValueError(f'Not found storage path of estimator: {stub_path}')

        with fs.open(stub_path, 'rb') as f:
            stub = pickle.load(f)

        model = DeepTable.load(model_path)
        stub.model = model

        return stub

    def get_iteration_scores(self):
        return []

    def __getstate__(self):
        try:
            state = super().__getstate__()
        except AttributeError:
            state = self.__dict__.copy()

        state['model'] = None

        return state


class HyperDT(HyperModel):
    def __init__(self, searcher, dispatcher=None, callbacks=[], reward_metric=None, discriminator=None,
                 max_model_size=0, cache_preprocessed_data=False, **config_kwargs):
        self.config_kwargs = config_kwargs
        metrics = config_kwargs.get('metrics')
        if metrics is None and reward_metric is None:
            raise ValueError('Must specify `reward_metric` or `metrics`.')
        if reward_metric is None:
            reward_metric = metrics[0]
        if metrics is None:
            metrics = [reward_metric]
            config_kwargs['metrics'] = metrics
        if reward_metric not in metrics:
            metrics.append(reward_metric)
            config_kwargs['metrics'] = metrics
        self.cache_preprocessed_data = cache_preprocessed_data
        HyperModel.__init__(self, searcher, dispatcher=dispatcher, callbacks=callbacks, reward_metric=reward_metric)

    def load_estimator(self, model_file):
        assert model_file is not None
        return DTEstimator.load(model_file)

    def _get_estimator(self, space_sample):
        estimator = DTEstimator(space_sample, self.cache_preprocessed_data, **self.config_kwargs)
        return estimator

    def export_trial_configuration(self, trial):
        default_conf = ModelConfig()
        new_conf = trial.space_sample.DT_Module.config
        conf_set = []
        for f in default_conf._fields:
            if new_conf.__getattribute__(f) != default_conf.__getattribute__(f):
                conf_set.append(f'\n\t{f}={new_conf.__getattribute__(f)}')
        str = f'ModelConfig({",".join(conf_set)})\n\nfit params:{trial.space_sample.fit_params.param_values}'
        return str


def default_dt_space(**hyperparams):
    space = HyperSpace()
    with space.as_default():
        p_nets = MultipleChoice(
            ['dnn_nets', 'linear', 'cin_nets', 'fm_nets', 'afm_nets', 'pnn_nets',
             'cross_nets', 'cross_dnn_nets', 'dcn_nets',
             'autoint_nets', 'fgcnn_dnn_nets', 'fibi_dnn_nets'], num_chosen_most=3)
        dt_module = DTModuleSpace(
            nets=p_nets,
            auto_categorize=Bool(),
            cat_remain_numeric=Bool(),
            auto_discrete=Bool(),
            apply_gbm_features=Bool(),
            gbm_feature_type=Choice([DT_consts.GBM_FEATURE_TYPE_DENSE, DT_consts.GBM_FEATURE_TYPE_EMB]),
            embeddings_output_dim=Choice([4, 10, 20]),
            embedding_dropout=Choice([0, 0.1, 0.2, 0.3, 0.4, 0.5]),
            stacking_op=Choice([DT_consts.STACKING_OP_ADD, DT_consts.STACKING_OP_CONCAT]),
            output_use_bias=Bool(),
            apply_class_weight=Bool(),
            earlystopping_patience=Choice([1, 3, 5])
        )
        dnn = DnnModule()(dt_module)
        fit = DTFit(**hyperparams)(dt_module)

    return space


def mini_dt_space(**hyperparams):
    space = HyperSpace()
    with space.as_default():
        p_nets = MultipleChoice(
            ['dnn_nets', 'linear', 'fm_nets'], num_chosen_most=2)
        dt_module = DTModuleSpace(
            nets=p_nets,
            auto_categorize=Bool(),
            cat_remain_numeric=Bool(),
            auto_discrete=Bool(),
            apply_gbm_features=Bool(),
            gbm_feature_type=Choice([DT_consts.GBM_FEATURE_TYPE_DENSE, DT_consts.GBM_FEATURE_TYPE_EMB]),
            embeddings_output_dim=Choice([4, 10]),
            embedding_dropout=Choice([0, 0.5]),
            stacking_op=Choice([DT_consts.STACKING_OP_ADD, DT_consts.STACKING_OP_CONCAT]),
            output_use_bias=Bool(),
            apply_class_weight=Bool(),
            earlystopping_patience=Choice([1, 3, 5])
        )
        dnn = DnnModule(hidden_units=Choice([100, 200]),
                        reduce_factor=Choice([1, 0.8]),
                        dnn_dropout=Choice([0, 0.3]),
                        use_bn=Bool(),
                        dnn_layers=2,
                        activation='relu')(dt_module)
        fit = DTFit(**hyperparams)(dt_module)

    return space


def mini_dt_space_validator(sample):
    nets = [p.value for p in sample.get_assigned_params() if p.alias.endswith('.nets')][0]
    return nets != ['fm_nets']


def tiny_dt_space(**hyperparams):
    space = HyperSpace()
    with space.as_default():
        dt_module = DTModuleSpace(
            nets=['dnn_nets'],
            auto_categorize=Bool(),
            cat_remain_numeric=Bool(),
            auto_discrete=False,
            apply_gbm_features=False,
            stacking_op=Choice([DT_consts.STACKING_OP_ADD, DT_consts.STACKING_OP_CONCAT]),
            output_use_bias=Bool(),
            apply_class_weight=Bool(),
            earlystopping_patience=Choice([1, 3, 5])
        )
        dnn = DnnModule(hidden_units=Choice([10, 20]),
                        reduce_factor=1,
                        dnn_dropout=Choice([0, 0.3]),
                        use_bn=False,
                        dnn_layers=2,
                        activation='relu')(dt_module)
        hyperparams['batch_size'] = [64, 100]
        fit = DTFit(**hyperparams)(dt_module)

    return space

    # categorical_columns='auto',
    # exclude_columns=[],
    # pos_label=None,
    # metrics=['accuracy'],
    # auto_categorize=False,
    # cat_exponent=0.5,
    # cat_remain_numeric=True,
    # auto_encode_label=True,
    # auto_imputation=True,
    # auto_discrete=False,
    # apply_gbm_features=False,
    # gbm_params={},
    # gbm_feature_type=DT_consts.GBM_FEATURE_TYPE_EMB,  # embedding/dense
    # fixed_embedding_dim=True,
    # embeddings_output_dim=4,
    # embeddings_initializer='uniform',
    # embeddings_regularizer=None,
    # embeddings_activity_regularizer=None,
    # dense_dropout=0,
    # embedding_dropout=0.3,
    # stacking_op=DT_consts.STACKING_OP_ADD,
    # output_use_bias=True,
    # apply_class_weight=False,
    # optimizer='auto',
    # loss='auto',
    # dnn_params={
    #     'hidden_units': ((128, 0, False), (64, 0, False)),
    #     'dnn_activation': 'relu',
    # },
    # autoint_params={
    #     'num_attention': 3,
    #     'num_heads': 1,
    #     'dropout_rate': 0,
    #     'use_residual': True,
    # },
    # fgcnn_params={'fg_filters': (14, 16),
    #               'fg_heights': (7, 7),
    #               'fg_pool_heights': (2, 2),
    #               'fg_new_feat_filters': (2, 2),
    #               },
    # fibinet_params={
    #     'senet_pooling_op': 'mean',
    #     'senet_reduction_ratio': 3,
    #     'bilinear_type': 'field_interaction',
    # },
    # cross_params={
    #     'num_cross_layer': 4,
    # },
    # pnn_params={
    #     'outer_product_kernel_type': 'mat',
    # },
    # afm_params={
    #     'attention_factor': 4,
    #     'dropout_rate': 0
    # },
    # cin_params={
    #     'cross_layer_size': (128, 128),
    #     'activation': 'relu',
    #     'use_residual': False,
    #     'use_bias': False,
    #     'direct': False,
    #     'reduce_D': False,
    # },
    # home_dir=None,
    # monitor_metric=None,
    # earlystopping_patience=1,
    # gpu_usage_strategy=DT_consts.GPU_USAGE_STRATEGY_GROWTH,
    # distribute_strategy=None,


def make_experiment(train_data,
                    searcher=None,
                    search_space=None,
                    **kwargs):
    """
    Utility to make CompeteExperiment instance with HyperDT.

    Parameters
    ----------

    Returns
    -------
    Runnable experiment object

    Notes:
    -------
    Initlialize Dask default client to enable dask in experiment.

    Examples:
    -------
    Create experiment with csv data file '/opt/data01/test.csv', and run it
    >>> experiment = make_experiment('/opt/data01/test.csv', target='y')
    >>> estimator = experiment.run()

    Create experiment with csv data file '/opt/data01/test.csv' with INFO logging, and run it
    >>> experiment = make_experiment('/opt/data01/test.csv', target='y', log_level='info')
    >>> estimator = experiment.run()

    Create experiment with parquet data files '/opt/data02/*.parquet', and run it with Dask
    >>> from dask.distributed import Client
    >>>
    >>> client = Client()
    >>> experiment = make_experiment('/opt/data02/*.parquet', target='y')
    >>> estimator = experiment.run()

    """

    searcher_options = kwargs.pop('searcher_options', {})
    if (searcher is None or isinstance(searcher, str)) and search_space is None:
        search_space = mini_dt_space
        searcher_options['space_sample_validation_fn'] = mini_dt_space_validator

    default_settings = dict(verbose=0,
                            # n_jobs=-1,
                            )
    for k, v in default_settings.items():
        if k not in kwargs.keys():
            kwargs[k] = v
    if kwargs.get('cv', True) and 'n_jobs' not in kwargs.keys():
        kwargs['n_jobs'] = -1

    config_options = {}
    option_keys = set(f for f in ModelConfig._fields if f not in {'name', 'task', 'metrics', 'nets', 'pos_label'})
    for k in option_keys:
        if k in kwargs.keys():
            config_options[k] = kwargs.pop(k)
    if 'pos_label' in kwargs.keys():
        config_options['pos_label'] = kwargs.get('pos_label')

    if isnotebook() and 'callbacks' not in kwargs.keys():
        from hypernets.experiment import SimpleNotebookCallback
        from hypernets.core import NotebookCallback as SearchNotebookCallback

        kwargs['callbacks'] = [SimpleNotebookCallback()]
        kwargs['search_callbacks'] = [SearchNotebookCallback()]

    experiment = _make_experiment(HyperDT, train_data,
                                  searcher=searcher,
                                  searcher_options=searcher_options,
                                  search_space=search_space,
                                  hyper_model_options=config_options,
                                  **kwargs)
    return experiment


_search_space_doc = """
    default is mini_dt_space."""


def _merge_doc():
    my_doc = DocLens(make_experiment.__doc__)
    params = DocLens(_make_experiment.__doc__).parameters
    params.pop('hyper_model_cls')
    params['search_space'] += _search_space_doc
    my_doc.parameters = params

    make_experiment.__doc__ = my_doc.render()


_merge_doc()
