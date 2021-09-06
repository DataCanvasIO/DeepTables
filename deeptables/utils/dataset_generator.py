# -*- coding:utf-8 -*-
"""

"""
from concurrent.futures import ThreadPoolExecutor
from distutils.version import LooseVersion
from functools import partial

import dask
import dask.dataframe as dd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical as tf_to_categorical

from deeptables.utils import consts, dt_logging

logger = dt_logging.get_logger(__name__)

TFDG_DASK_CHUNK = 100


class TFDatasetGenerator(object):
    def __init__(self, config, task, num_classes, *, categorical_columns, continuous_columns,
                 var_len_categorical_columns):
        self.distributed = config.distribute_strategy is not None
        self.task = task
        self.num_classes = num_classes
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        self.var_len_categorical_columns = var_len_categorical_columns

        super(TFDatasetGenerator, self).__init__()

    def __call__(self, X, y=None, *, batch_size, shuffle, drop_remainder):
        raise NotImplementedError()


class _TFDGForPandas(TFDatasetGenerator):
    def __call__(self, X, y=None, *, batch_size, shuffle, drop_remainder):
        train_data = {}
        # add categorical data
        if self.categorical_columns is not None and len(self.categorical_columns) > 0:
            train_data['input_categorical_vars_all'] = \
                X[[c.name for c in self.categorical_columns]].values.astype(consts.DATATYPE_TENSOR_FLOAT)

        # add continuous data
        if self.continuous_columns is not None and len(self.continuous_columns) > 0:
            for c in self.continuous_columns:
                train_data[c.name] = X[c.column_names].values.astype(consts.DATATYPE_TENSOR_FLOAT)

        # add var len categorical data
        if self.var_len_categorical_columns is not None and len(self.var_len_categorical_columns) > 0:
            for col in self.var_len_categorical_columns:
                train_data[col.name] = np.array(X[col.name].tolist())

        if y is None:
            ds = tf.data.Dataset.from_tensor_slices(train_data)
        else:
            y = np.array(y)
            if self.task == consts.TASK_MULTICLASS:
                y = tf_to_categorical(y, num_classes=self.num_classes)
            ds = tf.data.Dataset.from_tensor_slices((train_data, y))

        if shuffle:
            ds = ds.shuffle(buffer_size=X.shape[0])

        if self.distributed:
            # options = tf.data.Options()
            # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            ds = ds.repeat().batch(batch_size, drop_remainder=drop_remainder)  # .with_options(options=options)
        else:
            ds = ds.batch(batch_size, drop_remainder=drop_remainder and X.shape[0] >= batch_size)

        return ds


class _TFDGForDask(TFDatasetGenerator):
    def _get_meta(self, X):
        meta = {}
        col2idx = {c: i for i, c in enumerate(X.columns.to_list())}

        # add categorical data
        if self.categorical_columns is not None and len(self.categorical_columns) > 0:
            meta['input_categorical_vars_all'] = \
                (consts.DATATYPE_TENSOR_FLOAT, [col2idx[c.name] for c in self.categorical_columns])

        # add continuous data
        if self.continuous_columns is not None and len(self.continuous_columns) > 0:
            for cols in self.continuous_columns:
                meta[cols.name] = (consts.DATATYPE_TENSOR_FLOAT, [col2idx[c] for c in cols.column_names])

        # add var len categorical data
        if self.var_len_categorical_columns is not None and len(self.var_len_categorical_columns) > 0:
            for c in self.var_len_categorical_columns:
                # train_data[col.name] = np.array(X[col.name].tolist())
                meta[c.name] = (None, col2idx[c.name])

        return meta

    def __call__(self, X, y=None, *, batch_size, shuffle, drop_remainder):
        if LooseVersion(tf.version.VERSION) < LooseVersion('2.4.0'):
            ds = self._to_ds20(X, y, batch_size=batch_size, shuffle=shuffle, drop_remainder=drop_remainder)
        else:
            ds = self._to_ds24(X, y, batch_size=batch_size, shuffle=shuffle, drop_remainder=drop_remainder)
        return ds

    def _to_ds20(self, X, y=None, *, batch_size, shuffle, drop_remainder):
        ds_types = {}
        ds_shapes = {}
        meta = self._get_meta(X)
        for k, (dtype, idx) in meta.items():
            if dtype is not None:
                ds_shapes[k] = (None, len(idx))
                ds_types[k] = dtype
            else:  # var len
                v = X[k].head(1).tolist()[0]
                ds_shapes[k] = (None, len(v))
                ds_types[k] = 'int32'

        if y is not None:
            if isinstance(y, dd.Series):
                y = y.to_dask_array(lengths=True)
            if self.task == consts.TASK_MULTICLASS:
                y = self._to_categorical(y, num_classes=self.num_classes)
                ds_shape_y = None, self.num_classes
            else:
                ds_shape_y = None,
            ds_shapes = ds_shapes, ds_shape_y
            ds_types = ds_types, y.dtype

        X = X.to_dask_array(lengths=True)
        X, y = dask.persist(X, y)
        gen = partial(self._generate, meta, X, y,
                      batch_size=batch_size, shuffle=shuffle, drop_remainder=drop_remainder)
        ds = tf.data.Dataset.from_generator(gen, output_shapes=ds_shapes, output_types=ds_types)

        return ds

    def _to_ds24(self, X, y=None, *, batch_size, shuffle, drop_remainder):
        def to_spec(name, dtype, idx):
            if dtype is not None:
                spec = tf.TensorSpec(shape=(None, len(idx)), dtype=dtype)
            else:  # var len
                v = X[name].head(1).tolist()[0]
                spec = tf.TensorSpec(shape=(None, len(v)), dtype='int32')
            return spec

        meta = self._get_meta(X)
        sig = {k: to_spec(k, dtype, idx) for k, (dtype, idx) in meta.items()}

        if y is not None:
            if isinstance(y, dd.Series):
                y = y.to_dask_array(lengths=True)
            if self.task == consts.TASK_MULTICLASS:
                y = self._to_categorical(y, num_classes=self.num_classes)
                sig = sig, tf.TensorSpec(shape=(None, self.num_classes), dtype=y.dtype)
            else:
                sig = sig, tf.TensorSpec(shape=(None,), dtype=y.dtype)

        X = X.to_dask_array(lengths=True)
        X, y = dask.persist(X, y)
        gen = partial(self._generate, meta, X, y,
                      batch_size=batch_size, shuffle=shuffle, drop_remainder=drop_remainder)
        ds = tf.data.Dataset.from_generator(gen, output_signature=sig)

        return ds

    @staticmethod
    def _generate(meta, X, y, *, batch_size, shuffle, drop_remainder):
        total_size = dask.compute(X.shape)[0][0]
        chunk_size = min(total_size, batch_size * TFDG_DASK_CHUNK)
        fn = partial(_TFDGForDask._compute_chunk, X, y, chunk_size)
        chunks = _TFDGForDask._range(0, total_size, chunk_size, shuffle)
        ec = ThreadPoolExecutor(max_workers=2)

        try:
            for i_chunk, (X_chunk, y_chunk) in ec.map(fn, chunks):
                batch_stop = X_chunk.shape[0]
                if drop_remainder and total_size > batch_size:
                    batch_stop = batch_stop // batch_size * batch_size
                batches = _TFDGForDask._range(0, batch_stop, batch_size, shuffle)
                for bi in batches:
                    X_batch = X_chunk[bi:bi + batch_size]
                    data = {}
                    for k, (dtype, idx) in meta.items():
                        v = X_batch[:, idx]
                        if dtype is not None:
                            v = v.astype(dtype)
                        else:
                            v = np.array(v.tolist()).astype('int32')
                        data[k] = v
                    if y is not None:
                        y_batch = y_chunk[bi:bi + batch_size]
                        data = data, y_batch
                    yield data
        except GeneratorExit:
            pass
        except:
            import traceback
            traceback.print_exc()
            pass
        finally:
            ec.shutdown()
            del ec

    @staticmethod
    def _to_categorical(y, *, num_classes):
        if len(y.shape) == 1:
            y = y.reshape(dask.compute(y.shape[0])[0], 1)
        fn = partial(tf_to_categorical, num_classes=num_classes, dtype='float32')
        y = y.map_blocks(fn, dtype='float32')
        return y

    @staticmethod
    def _compute_chunk(X, y, chunk_size, i):
        try:
            Xc = X[i:i + chunk_size]
            yc = y[i:i + chunk_size] if y is not None else None
            r = dask.compute(Xc, yc)
        except:
            import traceback
            traceback.print_exc()
            r = None, None
        return i, r

    @staticmethod
    def _range(start, stop, step, shuffle):
        r = range(start, stop, step)
        if shuffle:
            import random
            r = list(r)
            random.shuffle(r)
        return r


def to_dataset(config, task, num_classes, X, y=None, *,
               batch_size, shuffle, drop_remainder,
               categorical_columns, continuous_columns, var_len_categorical_columns):
    cls = _TFDGForDask if isinstance(X, dd.DataFrame) else _TFDGForPandas
    logger.info(f'create dataset generator with {cls.__name__}, '
                f'batch_size={batch_size}, shuffle={shuffle}, drop_remainder={drop_remainder}')

    dg = cls(config, task, num_classes, categorical_columns=categorical_columns,
             continuous_columns=continuous_columns,
             var_len_categorical_columns=var_len_categorical_columns)
    ds = dg(X, y, batch_size=batch_size, shuffle=shuffle, drop_remainder=drop_remainder)
    return ds
