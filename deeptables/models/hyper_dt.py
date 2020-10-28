# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from deeptables.models.config import ModelConfig
from deeptables.models.deeptable import DeepTable
from deeptables.models.preprocessor import DefaultPreprocessor
from deeptables.utils import consts as DT_consts
from hypernets.core.search_space import HyperSpace, ModuleSpace, Choice, Bool, MultipleChoice
from hypernets.model.estimator import Estimator
from hypernets.model.hyper_model import HyperModel


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
    def __init__(self, batch_size=128, epochs=None, space=None, name=None, **hyperparams):
        if batch_size is None:
            batch_size = Choice([128, 256, 512])
        hyperparams['batch_size'] = batch_size

        if epochs is not None:
            hyperparams['epochs'] = epochs

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
            hidden_units = Choice([100, 200, 300, 500, 800, 1000])
        hyperparams['hidden_units'] = hidden_units

        if reduce_factor is None:
            reduce_factor = Choice([1, 0.8, 0.5])
        hyperparams['reduce_factor'] = reduce_factor

        if dnn_dropout is None:
            dnn_dropout = Choice([0, 0.1, 0.3, 0.5])
        hyperparams['dnn_dropout'] = dnn_dropout

        if use_bn is None:
            use_bn = Bool()
        hyperparams['use_bn'] = use_bn

        if dnn_layers is None:
            dnn_layers = Choice([1, 2, 3])
        hyperparams['dnn_layers'] = dnn_layers

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
    def __init__(self, space_sample, cache_preprocessed_data=False, cache_home=None, **config_kwargs):
        self.config_kwargs = config_kwargs
        self.cache_preprocessed_data = cache_preprocessed_data
        self.cache_home = cache_home
        self.model = self._build_model(space_sample)
        Estimator.__init__(self, space_sample=space_sample)

    def _build_model(self, space_sample):
        config = space_sample.DT_Module.config._replace(**self.config_kwargs)
        if self.cache_preprocessed_data:
            preprocessor = DefaultPreprocessor(config, cache_home=self.cache_home, use_cache=True)
        else:
            preprocessor = None
        model = DeepTable(config, preprocessor=preprocessor)
        return model

    def summary(self):
        try:
            mi = self.model.get_model()
            if mi is not None:
                mi.model.summary()
        except(Exception) as ex:
            print('---------no summary-------------')
            print(ex)

    def fit(self, X, y, **kwargs):
        fit_params = self.space_sample.__dict__.get('fit_params')
        if fit_params is not None:
            kwargs.update(fit_params.param_values)
        if kwargs.get('cross_validation') is not None:
            kwargs.pop('cross_validation')
            self.model.fit_cross_validation(X, y, **kwargs)
        else:
            self.model.fit(X, y, **kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def evaluate(self, X, y, metrics=None, **kwargs):
        result = self.model.evaluate(X, y, **kwargs)
        return result

    def predict_proba(self, X, **kwargs):
        result = self.model.predict_proba(X, **kwargs)
        return result

    def save(self, model_file):
        self.model.save(model_file)

    @staticmethod
    def load(model_file):
        return DeepTable.load(model_file)

class HyperDT(HyperModel):
    def __init__(self, searcher, dispatcher=None, callbacks=[], reward_metric=None, max_model_size=0,
                 cache_preprocessed_data=False, cache_home=None, **config_kwargs):
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
        self.cache_home = cache_home
        HyperModel.__init__(self, searcher, dispatcher=dispatcher, callbacks=callbacks, reward_metric=reward_metric)

    def _get_estimator(self, space_sample):
        estimator = DTEstimator(space_sample, self.cache_preprocessed_data, self.cache_home, **self.config_kwargs)
        return estimator

    def export_trail_configuration(self, trail):
        default_conf = ModelConfig()
        new_conf = trail.space_sample.DT_Module.config
        conf_set = []
        for f in default_conf._fields:
            if new_conf.__getattribute__(f) != default_conf.__getattribute__(f):
                conf_set.append(f'\n\t{f}={new_conf.__getattribute__(f)}')
        str = f'ModelConfig({",".join(conf_set)})\n\nfit params:{trail.space_sample.fit_params.param_values}'
        return str


def default_dt_space():
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
        fit = DTFit(batch_size=Choice([128, 256]))(dt_module)

    return space


def mini_dt_space():
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
        fit = DTFit(batch_size=Choice([128, 256]))(dt_module)

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
