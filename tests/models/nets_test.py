# -*- coding:utf-8 -*-

from sklearn.model_selection import train_test_split

from deeptables.models import deeptable, deepnets
from deeptables.datasets import dsutils
from tensorflow.keras import layers


class Test_DeepTable:

    def run_nets(self, nets):
        df_train = dsutils.load_adult().head(100)
        self.y = df_train.pop(14).values
        self.X = df_train

        conf = deeptable.ModelConfig(nets=nets,
                                     metrics=['AUC'],
                                     fixed_embedding_dim=True,
                                     embeddings_output_dim=2,
                                     apply_gbm_features=False,
                                     apply_class_weight=True)
        self.dt = deeptable.DeepTable(config=conf)

        self.X_train, \
        self.X_test, \
        self.y_train, \
        self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model, self.history = self.dt.fit(self.X_train, self.y_train, epochs=1)
        result = self.dt.evaluate(self.X_test, self.y_test)
        return result

    def test_DeepFM(self):
        result = self.run_nets(nets=deepnets.DeepFM)
        assert result['AUC'] >= 0.0

    def test_xDeepFM(self):
        result = self.run_nets(nets=deepnets.xDeepFM)
        assert result['AUC'] >= 0.0

    def test_WideDeep(self):
        result = self.run_nets(nets=deepnets.WideDeep)
        assert result['AUC'] >= 0.0

    def test_callable(self):
        result = self.run_nets(nets=[deepnets.linear])
        assert result['AUC'] >= 0.0

    def test_custom_nets(self):
        def custom_net(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
            out = layers.Dense(10)(flatten_emb_layer)
            return out

        result = self.run_nets(nets=[deepnets.linear, custom_net, 'dnn_nets'])
        assert result['AUC'] >= 0.0

    def test_linear(self):
        result = self.run_nets(nets=['linear'])
        assert result['AUC'] >= 0.0

    def test_afm(self):
        result = self.run_nets(nets=['afm_nets'])
        assert result['AUC'] >= 0.0

    def test_cin(self):
        result = self.run_nets(nets=['cin_nets'])
        assert result['AUC'] >= 0.0

    def test_pnn_nets(self):
        result = self.run_nets(nets=['opnn_nets', 'ipnn_nets', 'pnn_nets'])
        assert result['AUC'] >= 0.0

    def test_autoint_nets(self):
        result = self.run_nets(nets=['autoint_nets'])
        assert result['AUC'] >= 0.0

    def test_fgcnn_nets(self):
        result1 = self.run_nets(nets=['fg_nets'])
        result2 = self.run_nets(nets=['fgcnn_cin_nets'])
        result3 = self.run_nets(nets=['fgcnn_ipnn_nets'])
        result4 = self.run_nets(nets=['fgcnn_dnn_nets'])
        result5 = self.run_nets(nets=['fgcnn_fm_nets'])
        result6 = self.run_nets(nets=['fgcnn_afm_nets'])

        assert result1['AUC'] >= 0.0
        assert result2['AUC'] >= 0.0
        assert result3['AUC'] >= 0.0
        assert result4['AUC'] >= 0.0
        assert result5['AUC'] >= 0.0
        assert result6['AUC'] >= 0.0

    def test_fibi_nets(self):
        result1 = self.run_nets(nets=['fibi_nets'])
        result2 = self.run_nets(nets=['fibi_dnn_nets'])
        assert result1['AUC'] >= 0.0
        assert result2['AUC'] >= 0.0

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
        assert 14
