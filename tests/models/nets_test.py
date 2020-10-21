# -*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split

from deeptables.models import deeptable, deepnets
from deeptables.datasets import dsutils
from deeptables.models.layers import register_custom_objects
from tensorflow.keras import layers

from tensorflow.keras import backend as K
import tensorflow as tf
import multiprocessing
import tempfile, os, uuid
from multiprocessing import Manager


class Test_DeepTable:

    def run_load_model(self, p, X_test, y_test):
        model = deeptable.DeepTable.load(p)
        result_dt_loaded = model.evaluate(X_test, y_test)
        assert result_dt_loaded['AUC'] >= 0.0

    def run_nets(self, nets, **kwargs):
        df_train = dsutils.load_adult().head(100)
        y = df_train.pop(14).values
        X = df_train

        conf = deeptable.ModelConfig(nets=nets,
                                     metrics=['AUC'],
                                     fixed_embedding_dim=True,
                                     embeddings_output_dim=2,
                                     apply_gbm_features=False,
                                     apply_class_weight=True,
                                     **kwargs)

        dt = deeptable.DeepTable(config=conf)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model, history = dt.fit(X_train, y_train, epochs=1)

        result = dt.evaluate(X_test, y_test)
        assert result['AUC'] >= 0.0

        # test reload from disk
        # model_path = os.path.join("/tmp/dt_model", str(uuid.uuid4()))
        # dt.save(model_path)
        #
        # p = multiprocessing.Process(target=self.run_load_model, args=(model_path, X_test, y_test, ))
        # p.start()
        # p.join()

        return dt, result

    def test_all_nets(self):
        self.run_nets(
            nets=['dnn_nets', 'linear', 'cin_nets', 'fm_nets', 'afm_nets', 'opnn_nets', 'ipnn_nets', 'pnn_nets',
                  'cross_nets', 'cross_dnn_nets', 'dcn_nets', 'autoint_nets', 'fg_nets', 'fgcnn_cin_nets',
                  'fgcnn_fm_nets', 'fgcnn_ipnn_nets', 'fgcnn_dnn_nets', 'fibi_nets', 'fibi_dnn_nets'])

    def test_DeepFM(self):
        self.run_nets(nets=deepnets.DeepFM)

    def test_xDeepFM(self):
        self.run_nets(nets=deepnets.xDeepFM)

    def test_WideDeep(self):
        self.run_nets(nets=deepnets.WideDeep)

    def test_callable(self):
        self.run_nets(nets=[deepnets.linear])

    def test_custom_nets(self):
        def custom_net(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
            out = layers.Dense(10)(flatten_emb_layer)
            return out

        self.run_nets(nets=[deepnets.linear, custom_net, 'dnn_nets'])

    def test_linear(self):
        self.run_nets(nets=['linear'])

    def test_afm(self):
        self.run_nets(nets=['afm_nets'])

    def test_cin(self):
        self.run_nets(nets=['cin_nets'])

    def test_pnn_nets(self):
        self.run_nets(nets=['opnn_nets', 'ipnn_nets', 'pnn_nets'])

    def test_autoint_nets(self):
        self.run_nets(nets=['autoint_nets'],
                      autoint_params={
                          'num_attention': 3,
                          'num_heads': 2,
                          'dropout_rate': 0,
                          'use_residual': True,
                      }, )

    def test_fgcnn_nets(self):
        self.run_nets(nets=['fg_nets'])
        self.run_nets(nets=['fgcnn_cin_nets'])
        self.run_nets(nets=['fgcnn_ipnn_nets'])
        self.run_nets(nets=['fgcnn_dnn_nets'])
        self.run_nets(nets=['fgcnn_fm_nets'])
        self.run_nets(nets=['fgcnn_afm_nets'])

    def test_fibi_nets(self):
        self.run_nets(nets=['fibi_nets'])
        self.run_nets(nets=['fibi_dnn_nets'])

    def test_custom_dnn(self):
        df_train = dsutils.load_adult().head(100)
        y = df_train.pop(14).values
        X = df_train

        conf = deeptable.ModelConfig(nets=['dnn_nets'],
                                     dnn_params={
                                         'custom_dnn_fn': deepnets.custom_dnn_D_A_D_B,
                                         'hidden_units': ((128, 0.2, True), (64, 0, False)),
                                     },
                                     metrics=['AUC'],
                                     fixed_embedding_dim=True,
                                     embeddings_output_dim=2,
                                     apply_gbm_features=False,
                                     apply_class_weight=True)
        dt = deeptable.DeepTable(config=conf)
        model, history = dt.fit(X, y, epochs=1)
        l1 = model.model.get_layer('dnn_custom_dense_1')
        l2 = model.model.get_layer('dnn_custom_dropout_1')
        l3 = model.model.get_layer('dnn_custom_bn_1')
        l4 = model.model.get_layer('dnn_custom_dense_2')

        assert l1
        assert l2
        assert l3
        assert l4

    def test_save_load_custom_nets(self):
        register_custom_objects(
            {
                'CustomFM': CustomFM,
            })

        def custom_net(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
            if embeddings is None or len(embeddings) <= 0:
                model_desc.add_net('fm', (None), (None))
                return None
            fm_concat = layers.Concatenate(axis=1, name='concat_fm_embedding')(embeddings)
            fm_output = CustomFM(name='fm_layer')(fm_concat)  # use custom layer
            model_desc.add_net('fm', fm_concat.shape, fm_output.shape)
            return fm_output

        dt, result = self.run_nets(nets=[custom_net, 'dnn_nets'])

        filepath = tempfile.mkdtemp()
        dt.save(filepath)
        # assert os.path.exists(f'{filepath}/dt.pkl')
        # assert os.path.exists(f'{filepath}/custom_net+dnn_nets.h5')

        newdt = deeptable.DeepTable.load(filepath)
        assert newdt.best_model


class CustomFM(layers.Layer):

    def __init__(self, **kwargs):
        super(CustomFM, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        if K.ndim(x) != 3:
            raise ValueError(f'Wrong dimensions of inputs, expected 3 but input {K.ndim(x)}.')
        square_of_sum = tf.square(tf.reduce_sum(
            x, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(
            x * x, axis=1, keepdims=True)
        cross = square_of_sum - sum_of_square
        cross = 0.5 * tf.reduce_sum(cross, axis=2, keepdims=False)
        return cross
