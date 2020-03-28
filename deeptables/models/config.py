# -*- coding:utf-8 -*-

import collections
import os
from ..utils import consts
from . import deepnets as deepnets


class ModelConfig(collections.namedtuple('ModelConfig',
                                         ['name',
                                          'nets',
                                          'categorical_columns',
                                          'exclude_columns',
                                          'pos_label',
                                          'metrics',
                                          'auto_categorize',
                                          'cat_exponent',
                                          'cat_remain_numeric',
                                          'auto_encode_label',
                                          'auto_imputation',
                                          'auto_discrete',
                                          'apply_gbm_features',
                                          'gbm_params',
                                          'gbm_feature_type',
                                          'fixed_embedding_dim',
                                          'embeddings_output_dim',
                                          'embeddings_initializer',
                                          'embeddings_regularizer',
                                          'embeddings_activity_regularizer',
                                          'dense_dropout',
                                          'embedding_dropout',
                                          'stacking_op',
                                          'output_use_bias',
                                          'apply_class_weight',
                                          'optimizer',
                                          'loss',
                                          'dnn_params',
                                          'autoint_params',
                                          'fgcnn_params',
                                          'fibinet_params',
                                          'cross_params',
                                          'pnn_params',
                                          'afm_params',
                                          'cin_params',
                                          'home_dir',
                                          'monitor_metric',
                                          'earlystopping_patience',
                                          'gpu_usage_strategy',
                                          'distribute_strategy',
                                          ])):
    def __hash__(self):
        return self.name.__hash__()

    def __new__(cls,
                name='conf-1',
                nets=['dnn_nets'],
                categorical_columns='auto',
                exclude_columns=[],
                pos_label=None,
                metrics=['accuracy'],
                auto_categorize=False,
                cat_exponent=0.5,
                cat_remain_numeric=True,
                auto_encode_label=True,
                auto_imputation=True,
                auto_discrete=False,
                apply_gbm_features=False,
                gbm_params={},
                gbm_feature_type=consts.GBM_FEATURE_TYPE_EMB,  # embedding/dense
                fixed_embedding_dim=True,
                embeddings_output_dim=4,
                embeddings_initializer='uniform',
                embeddings_regularizer=None,
                embeddings_activity_regularizer=None,
                dense_dropout=0,
                embedding_dropout=0.3,
                stacking_op=consts.STACKING_OP_ADD,
                output_use_bias=True,
                apply_class_weight=False,
                optimizer='auto',
                loss='auto',
                dnn_params={
                    'dnn_units': ((128, 0, False), (64, 0, False)),
                    'dnn_activation': 'relu',
                },
                autoint_params={
                    'num_attention': 3,
                    'num_heads': 1,
                    'dropout_rate': 0,
                    'use_residual': True,
                },
                fgcnn_params={'fg_filters': (14, 16),
                              'fg_heights': (7, 7),
                              'fg_pool_heights': (2, 2),
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
                home_dir=None,
                monitor_metric=None,
                earlystopping_patience=1,
                gpu_usage_strategy=consts.GPU_USAGE_STRATEGY_GROWTH,
                distribute_strategy=None,
                ):
        nets = deepnets.get_nets(nets)

        if home_dir is None and os.environ.get(consts.ENV_DEEPTABLES_HOME) is not None:
            home_dir = os.environ.get(consts.ENV_DEEPTABLES_HOME)

        return super(ModelConfig, cls).__new__(cls,
                                               name,
                                               nets,
                                               categorical_columns,
                                               exclude_columns,
                                               pos_label,
                                               metrics,
                                               auto_categorize,
                                               cat_exponent,
                                               cat_remain_numeric,
                                               auto_encode_label,
                                               auto_imputation,
                                               auto_discrete,
                                               apply_gbm_features,
                                               gbm_params,
                                               gbm_feature_type,
                                               fixed_embedding_dim,
                                               embeddings_output_dim,
                                               embeddings_initializer,
                                               embeddings_regularizer,
                                               embeddings_activity_regularizer,
                                               dense_dropout,
                                               embedding_dropout,
                                               stacking_op,
                                               output_use_bias,
                                               apply_class_weight,
                                               optimizer,
                                               loss,
                                               dnn_params,
                                               autoint_params,
                                               fgcnn_params,
                                               fibinet_params,
                                               cross_params,
                                               pnn_params,
                                               afm_params,
                                               cin_params,
                                               home_dir,
                                               monitor_metric,
                                               earlystopping_patience,
                                               gpu_usage_strategy,
                                               distribute_strategy,
                                               )
    @property
    def first_metric_name(self):
        if self.metrics is None or len(self.metrics)<=0:
            raise ValueError('`metrics` is none or empty.')
        first_metric = self.metrics[0]
        if isinstance(first_metric, str):
            return first_metric
        if callable(first_metric):
            return first_metric.__name__
        raise ValueError('`metric` must be string or callable object.')