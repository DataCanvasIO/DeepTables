# -*- coding:utf-8 -*-
"""
DeepTables constants module.
"""

from hypernets.utils.const import *

PROJECT_NAME = 'deeptables'

# TASK_AUTO = 'auto'
# TASK_BINARY = 'binary'
# TASK_MULTICLASS = 'multiclass'
# TASK_REGRESSION = 'regression'
# TASK_MULTILABEL = 'multilabel'

INPUT_PREFIX_CAT = 'cat_'
INPUT_PREFIX_NUM = 'input_continuous_'
INPUT_PREFIX_SEQ = 'seq_'
LAYER_PREFIX_EMBEDDING = 'emb_'

# COLUMNNAME_POSTFIX_DISCRETE = '_discrete'
# COLUMNNAME_POSTFIX_CATEGORIZE = '_cat'

DATATYPE_TENSOR_FLOAT = 'float32'
DATATYPE_PREDICT_CLASS = 'int32'
# DATATYPE_LABEL = 'int16'

LAYER_NAME_BN_DENSE_ALL = 'bn_dense_all'
LAYER_NAME_CONCAT_CONT_INPUTS = 'concat_continuous_inputs'

MODEL_SELECT_MODE_MIN = 'min'
MODEL_SELECT_MODE_MAX = 'max'
MODEL_SELECT_MODE_AUTO = 'auto'

METRIC_NAME_AUC = 'AUC'
METRIC_NAME_ACCURACY = 'accuracy'
METRIC_NAME_MSE = 'mse'

MODEL_SELECTOR_BEST = 'best'
MODEL_SELECTOR_CURRENT = 'current'
MODEL_SELECTOR_ALL = 'all'

EMBEDDING_OUT_DIM_DEFAULT = 4

GBM_FEATURE_TYPE_EMB = 'embedding'
GBM_FEATURE_TYPE_DENSE = 'dense'

STACKING_OP_CONCAT = 'concat'
STACKING_OP_ADD = 'add'

GPU_USAGE_STRATEGY_GROWTH = 'memory_growth'

ENV_DEEPTABLES_HOME = 'DEEPTABLES_HOME'
