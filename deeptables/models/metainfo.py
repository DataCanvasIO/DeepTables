# -*- coding:utf-8 -*-

import collections
from ..utils import consts


# class DeepMeta:
#     def __init__(self):
#
#         self.categorical_columns
#         self.continuous_columns

# class Column:
#     def __init__(self, name, dtype, has_nans):
#         self.name = name
#         self.dtype = dtype
#         self.has_nans = has_nans
#
#
# class CategoricalColumn(Column):
#     def __init__(self, name, dtype, has_nans, num_uniques):
#         super(CategoricalColumn).__init__(name, dtype, has_nans)
#         self.num_uniques = num_uniques
#
#
# class ContinuousColumn(Column):
#     def __init__(self, name, dtype, has_nans, min, max):
#         super(ContinuousColumn).__init__(name, dtype, has_nans)
#         self.min = min
#         self.max = max


class CategoricalColumn(collections.namedtuple('CategoricalColumn',
                                               ['name',
                                                'vocabulary_size',
                                                'embeddings_output_dim',
                                                'dtype',
                                                'input_name',
                                                ])):
    def __hash__(self):
        return self.name.__hash__()

    def __new__(cls, name, vocabulary_size, embeddings_output_dim=10, dtype='int32', input_name=None, ):
        if input_name is None:
            input_name = consts.INPUT_PREFIX_CAT + name
        if embeddings_output_dim == 0:
            embeddings_output_dim = int(round(vocabulary_size ** 0.25))
        return super(CategoricalColumn, cls).__new__(cls, name, vocabulary_size, embeddings_output_dim, dtype,
                                                     input_name)


class ContinuousColumn(collections.namedtuple('ContinuousColumn',
                                              ['name',
                                               'column_names',
                                               'input_dim',
                                               'dtype',
                                               'input_name',
                                               ])):
    def __hash__(self):
        return self.name.__hash__()

    def __new__(cls, name, column_names, input_dim=0, dtype='float32', input_name=None, ):
        input_dim = len(column_names)
        return super(ContinuousColumn, cls).__new__(cls, name, column_names, input_dim, dtype, input_name)
