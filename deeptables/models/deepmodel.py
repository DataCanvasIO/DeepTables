# -*- coding:utf-8 -*-
import collections
import io
import math
from collections import OrderedDict
from typing import List

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Concatenate, Flatten, Input, Add, BatchNormalization, Dropout
from tensorflow.keras.models import Model, load_model, save_model

from deeptables.models.metainfo import CategoricalColumn
from hypernets.tabular import get_tool_box
from . import deepnets
from .layers import MultiColumnEmbedding, dt_custom_objects, VarLenColumnEmbedding
from .metainfo import VarLenCategoricalColumn
from ..utils import dt_logging, consts, gpu, to_dataset

logger = dt_logging.get_logger(__name__)


class DeepModel:
    """ Class for neural network models"""

    def __init__(self,
                 task,
                 num_classes,
                 config,
                 categorical_columns,
                 continuous_columns,
                 model_file=None,
                 var_categorical_len_columns=None, ):

        # set gpu usage strategy before build model
        if config.gpu_usage_strategy == consts.GPU_USAGE_STRATEGY_GROWTH:
            gpu.set_memory_growth()
        self.model_desc = ModelDesc()
        self.categorical_columns = categorical_columns
        self.continuous_columns = continuous_columns
        self.var_len_categorical_columns = var_categorical_len_columns
        self.task = task
        self.num_classes = num_classes
        self.config = config
        self.model_file = model_file
        self.model = None
        if model_file is not None:
            # fixme: `load_model` executed multiple times in a process,
            #  resulting in a metric name rename to like auc_1, auc_2
            self.model = self._load_model(model_file, dt_custom_objects)

    def fit(self, X=None, y=None, batch_size=128, epochs=1, verbose=1, callbacks=None,
            validation_split=0.2, validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None,
            initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_freq=1,
            max_queue_size=10, workers=1, use_multiprocessing=False):
        tb = get_tool_box(X)
        if validation_data is None:
            X, X_val, y, y_val = tb.train_test_split(X, y, test_size=validation_split)
        else:
            if len(validation_data) != 2:
                raise ValueError(f'Unexpected validation_data length, expected 2 but {len(validation_data)}.')
            X_val, y_val = validation_data[0], validation_data[1]

        if batch_size is None:
            batch_size = 128

        if steps_per_epoch is None:
            steps_per_epoch = len(X) // batch_size
            if steps_per_epoch == 0:
                steps_per_epoch = 1
        if validation_steps is None:
            validation_steps = len(X_val) // batch_size - 1
            if validation_steps <= 1:
                validation_steps = 1

        train_data = self.__get_train_data(X, y, batch_size=batch_size, shuffle=shuffle).repeat(count=epochs)
        validation_data = self.__get_train_data(X_val, y_val, batch_size=batch_size, shuffle=shuffle)

        if self.config.distribute_strategy is not None:
            from tensorflow.python.distribute.distribute_lib import Strategy
            if not isinstance(self.config.distribute_strategy, Strategy):
                raise ValueError(f'[distribute_strategy] in ModelConfig must be an instance of {Strategy}')
            options = tf.data.Options()
            options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
            train_data = train_data.with_options(options=options)
            validation_data = validation_data.with_options(options=options)
            with self.config.distribute_strategy.scope():
                self.model = self.__build_model(task=self.task,
                                                num_classes=self.num_classes,
                                                nets=self.config.nets,
                                                categorical_columns=self.categorical_columns,
                                                continuous_columns=self.continuous_columns,
                                                var_len_categorical_columns=self.var_len_categorical_columns,
                                                config=self.config)
        else:
            self.model = self.__build_model(task=self.task,
                                            num_classes=self.num_classes,
                                            nets=self.config.nets,
                                            categorical_columns=self.categorical_columns,
                                            continuous_columns=self.continuous_columns,
                                            var_len_categorical_columns=self.var_len_categorical_columns,
                                            config=self.config)

        logger.info(f'training...')
        history = self.model.fit(train_data,
                                 epochs=epochs,
                                 verbose=verbose,
                                 validation_data=validation_data,
                                 shuffle=shuffle,
                                 callbacks=callbacks,
                                 class_weight=class_weight,
                                 sample_weight=sample_weight,
                                 initial_epoch=initial_epoch,
                                 steps_per_epoch=steps_per_epoch,
                                 validation_steps=validation_steps,
                                 validation_freq=validation_freq,
                                 max_queue_size=max_queue_size,
                                 workers=workers,
                                 use_multiprocessing=use_multiprocessing,
                                 )
        logger.info(f'Training finished.')
        history.history = IgnoreCaseDict(history.history)  # update dict metrics
        return history

    def predict(self, X, batch_size=128, verbose=0):
        return self.__predict(self.model, X, batch_size=batch_size, verbose=verbose)

    def __predict(self, model, X, batch_size=128, verbose=0):
        logger.info("Performing predictions...")
        ds = self.__get_prediction_data(X, batch_size=batch_size)
        steps = math.ceil(len(X) / batch_size)
        return model.predict(ds, steps=steps, verbose=verbose)

    def apply(self, X, output_layers=[], concat_outputs=False, batch_size=128,
              verbose=0, transformer=None):
        model = self.__build_proxy_model(self.model, output_layers, concat_outputs)
        output = self.__predict(model, X, batch_size=batch_size, verbose=verbose)

        # support datasets transformation before returning, e.g. t-sne
        if transformer is None:
            return output
        else:
            if isinstance(output, list):
                output_t = []
                for i, x_o in enumerate(output):
                    if len(x_o.shape) > 2:
                        x_o = x_o.reshape((x_o.shape[0], -1))
                    if logger.is_info_enabled():
                        logger.info(f'Performing transformation on [{output_layers[i]}] by "{str(transformer)}", '
                                    f'input shape:{x_o.shape}.')
                    output_t.append(transformer.fit_transform(x_o))
                return output_t
            else:
                return transformer.fit_transform(output)

    def evaluate(self, X_test, y_test, batch_size=256, verbose=0, return_dict=True):
        logger.info("Performing evaluation...")
        ds = self.__get_prediction_data(X_test, y_test, batch_size=batch_size)
        steps = math.ceil(len(X_test) / batch_size)
        result = self.model.evaluate(ds, steps=steps, verbose=verbose)
        if return_dict:
            result = {k: v for k, v in zip(self.model.metrics_names, result)}
            return IgnoreCaseDict(result)
        else:
            return result

    @staticmethod
    def _load_model(filepath, custom_objects):
        import h5py
        from deeptables.utils import fs

        with fs.open(filepath, 'rb') as f:
            data = f.read()

        buf = io.BytesIO(data)
        del data
        with h5py.File(buf, 'r') as h:
            return load_model(h, custom_objects)

    def save(self, filepath):
        import h5py
        from deeptables.utils import fs

        with fs.open(filepath, 'wb') as f:
            buf = io.BytesIO()
            with h5py.File(buf, 'w') as h:
                save_model(self.model, h, save_format='h5')
            data = buf.getvalue()
            buf.close()
            f.write(data)

    def release(self):
        del self.model
        self.model = None
        K.clear_session()

    def __get_train_data(self, X, y, *, batch_size, shuffle):
        ds = to_dataset(self.config, self.task, self.num_classes, X, y,
                        batch_size=batch_size, shuffle=shuffle, drop_remainder=True,
                        categorical_columns=self.categorical_columns,
                        continuous_columns=self.continuous_columns,
                        var_len_categorical_columns=self.var_len_categorical_columns)
        return ds

    def __get_prediction_data(self, X, y=None, *, batch_size):
        ds = to_dataset(self.config, self.task, self.num_classes, X, y,
                        batch_size=batch_size, shuffle=False, drop_remainder=False,
                        categorical_columns=self.categorical_columns,
                        continuous_columns=self.continuous_columns,
                        var_len_categorical_columns=self.var_len_categorical_columns)

        return ds

    def __build_proxy_model(self, model, output_layers=[], concat_output=False):
        model.trainable = False
        if len(output_layers) <= 0:
            raise ValueError('"output_layers" at least 1 element.')
        outputs = [model.get_layer(l).output for l in output_layers]

        if len(outputs) <= 0:
            raise ValueError(f'No layer found in the model:{output_layers}')
        if len(outputs) > 1 and concat_output:
            outputs = Concatenate()(outputs)
        proxy = Model(inputs=model.input, outputs=outputs)
        proxy.compile(optimizer=model.optimizer, loss=model.loss)
        return proxy

    def __build_model(self, task, num_classes, nets, categorical_columns, continuous_columns,
                      var_len_categorical_columns, config):
        logger.info(f'Building model...')
        self.model_desc = ModelDesc()
        categorical_inputs, continuous_inputs, var_len_categorical_inputs = \
            self.__build_inputs(categorical_columns, continuous_columns, var_len_categorical_columns)
        embeddings = self.__build_embeddings(categorical_columns, categorical_inputs, var_len_categorical_columns,
                                             var_len_categorical_inputs, config.embedding_dropout)
        dense_layer = self.__build_denses(continuous_columns, continuous_inputs, config.dense_dropout)

        flatten_emb_layer = None
        if len(embeddings) > 0:
            if len(embeddings) == 1:
                flatten_emb_layer = Flatten(name='flatten_embeddings')(embeddings[0])
            else:
                flatten_emb_layer = Flatten(name='flatten_embeddings')(
                    Concatenate(name='concat_embeddings_axis_0')(embeddings))

        self.model_desc.nets = nets
        self.model_desc.stacking = config.stacking_op
        concat_emb_dense = self.__concat_emb_dense(flatten_emb_layer, dense_layer)
        # concat_emb_dense = flatten_emb_layer
        outs = {}
        for net in nets:
            logit = deepnets.get(net)
            out = logit(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, self.config, self.model_desc)
            if out is not None:
                outs[net] = out
        if len(outs) > 1:
            logits = []
            for name, out in outs.items():
                if len(out.shape) > 2:
                    out = Flatten(name=f'flatten_{name}_out')(out)
                if out.shape[-1] > 1:
                    logit = Dense(1, use_bias=False, activation=None, name=f'dense_logit_{name}')(out)
                else:
                    logit = out
                logits.append(logit)
            if config.stacking_op == consts.STACKING_OP_ADD:
                x = Add(name='add_logits')(logits)
            elif config.stacking_op == consts.STACKING_OP_CONCAT:
                x = Concatenate(name='concat_logits')(logits)
            else:
                raise ValueError(f'Unsupported stacking_op:{config.stacking_op}.')
        elif (len(outs) == 1):
            name, out = outs.popitem()
            # out = list(outs.values())[0]
            if len(out.shape) > 2:
                out = Flatten(name=f'flatten_{name}_out')(out)
            x = out
        else:
            raise ValueError(f'Unexpected logit output.{outs}')
        all_inputs = list(categorical_inputs.values()) + list(var_len_categorical_inputs.values()) + \
                     list(continuous_inputs.values())
        output = self.__output_layer(x, task, num_classes, use_bias=self.config.output_use_bias)
        model = Model(inputs=all_inputs, outputs=output)
        model = self.__compile_model(model, task, num_classes, config.optimizer, config.loss, config.metrics)
        if logger.is_info_enabled():
            logger.info(self.model_desc)
        return model

    def __compile_model(self, model, task, num_classes, optimizer, loss, metrics):
        if optimizer == 'auto':
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        if loss == 'auto':
            if task == consts.TASK_BINARY or task == consts.TASK_MULTILABEL:
                loss = 'binary_crossentropy'
            elif task == consts.TASK_REGRESSION:
                loss = 'mse'
            elif task == consts.TASK_MULTICLASS:
                if num_classes == 2:
                    loss = 'binary_crossentropy'
                else:
                    loss = 'categorical_crossentropy'
        self.model_desc.optimizer = optimizer
        self.model_desc.loss = loss
        model.compile(optimizer, loss, metrics=metrics)
        return model

    def __concat_emb_dense(self, flatten_emb_layer, dense_layer):
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
        self.model_desc.set_concat_embed_dense(x.shape)
        return x

    def __build_inputs(self, categorical_columns: List[CategoricalColumn], continuous_columns,
                       var_len_categorical_columns: List[VarLenCategoricalColumn] = None):
        categorical_inputs = OrderedDict()
        var_len_categorical_inputs = OrderedDict()
        continuous_inputs = OrderedDict()

        if categorical_columns is not None and len(categorical_columns) > 0:
            categorical_inputs['all_categorical_vars'] = Input(shape=(len(categorical_columns),),
                                                               name='input_categorical_vars_all')
            self.model_desc.add_input('all_categorical_vars', len(categorical_columns))

        # make input for var len feature
        if var_len_categorical_columns is not None and len(var_len_categorical_columns) > 0:
            for col in var_len_categorical_columns:
                var_len_categorical_inputs[col.name] = Input(shape=(col.max_elements_length,), name=col.name)
                self.model_desc.add_input(col.name, col.max_elements_length)

        for column in continuous_columns:
            continuous_inputs[column.name] = Input(shape=(column.input_dim,), name=column.name,
                                                   dtype=column.dtype)
            self.model_desc.add_input(column.name, column.input_dim)

        return categorical_inputs, continuous_inputs, var_len_categorical_inputs

    def __construct_var_len_embedding(self, column: VarLenCategoricalColumn, var_len_inputs, embedding_dropout):
        input_layer = var_len_inputs[column.name]
        var_len_embeddings = VarLenColumnEmbedding(pooling_strategy=column.pooling_strategy,
                                                   input_dim=column.vocabulary_size,
                                                   output_dim=column.embeddings_output_dim,
                                                   dropout_rate=embedding_dropout,
                                                   name=consts.LAYER_PREFIX_EMBEDDING + column.name,
                                                   embeddings_initializer=self.config.embeddings_initializer,
                                                   embeddings_regularizer=self.config.embeddings_regularizer,
                                                   activity_regularizer=self.config.embeddings_activity_regularizer
                                                   )(input_layer)
        return var_len_embeddings

    def __build_embeddings(self, categorical_columns, categorical_inputs,
                           var_len_categorical_columns: List[VarLenCategoricalColumn], var_len_inputs,
                           embedding_dropout):
        if 'all_categorical_vars' in categorical_inputs:
            input_layer = categorical_inputs['all_categorical_vars']
            input_dims = [column.vocabulary_size for column in categorical_columns]
            output_dims = [column.embeddings_output_dim for column in categorical_columns]
            embeddings = MultiColumnEmbedding(input_dims, output_dims, embedding_dropout,
                                              name=consts.LAYER_PREFIX_EMBEDDING + 'categorical_vars_all',
                                              embeddings_initializer=self.config.embeddings_initializer,
                                              embeddings_regularizer=self.config.embeddings_regularizer,
                                              activity_regularizer=self.config.embeddings_activity_regularizer,
                                              )(input_layer)
            self.model_desc.set_embeddings(input_dims, output_dims, embedding_dropout)
        else:
            embeddings = []

        # do embedding for var len feature
        if var_len_categorical_columns is not None and len(var_len_categorical_columns) > 0:
            for c in var_len_categorical_columns:
                # todo add var len embedding description
                var_len_embedding = self.__construct_var_len_embedding(c, var_len_inputs, embedding_dropout)
                embeddings.append(var_len_embedding)

        return embeddings

    def __build_denses(self, continuous_columns, continuous_inputs, dense_dropout, use_batchnormalization=False):
        dense_layer = None
        if continuous_inputs:
            if len(continuous_inputs) > 1:
                dense_layer = Concatenate(name=consts.LAYER_NAME_CONCAT_CONT_INPUTS)(list(continuous_inputs.values()))
            else:
                dense_layer = list(continuous_inputs.values())[0]
        if dense_dropout > 0:
            dense_layer = Dropout(dense_dropout, name='dropout_dense_input')(dense_layer)
        if use_batchnormalization:
            dense_layer = BatchNormalization(name=consts.LAYER_NAME_BN_DENSE_ALL)(dense_layer)
        self.model_desc.set_dense(dense_dropout, use_batchnormalization)
        return dense_layer

    def __output_layer(self, x, task, num_classes, use_bias=True):
        if task == consts.TASK_BINARY:
            activation = 'sigmoid'
            output_dim = 1
        elif task == consts.TASK_REGRESSION:
            activation = None
            output_dim = 1
        elif task == consts.TASK_MULTICLASS:
            if num_classes:
                activation = 'softmax'
                output_dim = num_classes
            else:
                raise ValueError('"config.multiclass_classes" value must be provided for multi-class task.')
        elif task == consts.TASK_MULTILABEL:
            activation = 'sigmoid'
            output_dim = num_classes
        else:
            raise ValueError(f'Unknown task type:{task}')

        output = Dense(output_dim, activation=activation, name='task_output', use_bias=use_bias)(x)
        self.model_desc.set_output(activation, output.shape, use_bias)
        return output


class ModelDesc:
    def __init__(self):
        self.inputs = []
        self.embeddings = None
        self.dense = None
        self.concat_embed_dense = None
        self.nets = []
        self.nets_info = []
        self.stacking = None
        self.output = None
        self.loss = None
        self.optimizer = None

    def add_input(self, name, num_columns):
        self.inputs.append(f'{name}: ({num_columns})')

    def set_embeddings(self, input_dims, output_dims, embedding_dropout):
        self.embeddings = f'input_dims: {input_dims}\n' \
                          f'output_dims: {output_dims}\n' \
                          f'dropout: {embedding_dropout}'

    def set_dense(self, dense_dropout, use_batchnormalization):
        self.dense = f'dropout: {dense_dropout}\n' \
                     f'batch_normalization: {use_batchnormalization}'

    def set_concat_embed_dense(self, output_shape):
        self.concat_embed_dense = f'shape: {output_shape}'

    def add_net(self, name, input_shape, output_shape):
        self.nets_info.append(f'{name}: input_shape {input_shape}, output_shape {output_shape}')

    def set_output(self, activation, output_shape, use_bias):
        self.output = f'activation: {activation}, output_shape: {output_shape}, use_bias: {use_bias}'

    def nets_desc(self):
        return '\n'.join(self.nets_info)

    def optimizer_info(self):
        if self.optimizer is None:
            return None
        if hasattr(self.optimizer, '_name'):
            name = getattr(self.optimizer, '_name')
        else:
            name = self.optimizer
        return name

    def __str__(self):
        text = f'>>>>>>>>>>>>>>>>>>>>>> Model Desc <<<<<<<<<<<<<<<<<<<<<<< \n' \
               f'---------------------------------------------------------\n' \
               f'inputs:\n' \
               f'---------------------------------------------------------\n' \
               f'{[c for c in self.inputs]}\n' \
               f'---------------------------------------------------------\n' \
               f'embeddings:\n' \
               f'---------------------------------------------------------\n' \
               f'{self.embeddings}\n' \
               f'---------------------------------------------------------\n' \
               f'dense: {self.dense}\n' \
               f'---------------------------------------------------------\n' \
               f'concat_embed_dense: {self.concat_embed_dense}\n' \
               f'---------------------------------------------------------\n' \
               f'nets: {self.nets}\n' \
               f'---------------------------------------------------------\n' \
               f'{self.nets_desc()}\n' \
               f'---------------------------------------------------------\n' \
               f'stacking_op: {self.stacking}\n' \
               f'---------------------------------------------------------\n' \
               f'output: {self.output}\n' \
               f'loss: {self.loss}\n' \
               f'optimizer: {self.optimizer_info()}\n' \
               f'---------------------------------------------------------\n' \
               f''
        return text


class IgnoreCaseDict(collections.UserDict):

    def __init__(self, *args, **kwargs):
        super(IgnoreCaseDict, self).__init__(*args, **kwargs)
        # update key
        for k in self.data:
            if not isinstance(k, str):
                raise KeyError(f"Key should be str but is {k}")

        _data = {k.lower(): self.data[k] for k in self.data}
        self.data.update(_data)

    def __contains__(self, item):
        if not isinstance(item, str):
            raise KeyError(f"Key should be str but is {item}")
        return item.lower() in self.data

    def __setitem__(self, item, value):
        if not isinstance(item, str):
            raise KeyError(f"Key should be str but is {item}")
        self.data[item.lower()] = value

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise KeyError(f"Key should be str but is {item}")
        return self.data[item.lower()]
