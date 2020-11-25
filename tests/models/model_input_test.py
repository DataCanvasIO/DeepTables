# -*- encoding: utf-8 -*-
import pandas as pd
import pytest

from deeptables.datasets import dsutils
from deeptables.models import deeptable
from deeptables.utils import consts


class TestModelInput:

    def setup_class(cls):
        cls.df_bank = dsutils.load_bank().sample(frac=0.01)
        cls.df_movielens = dsutils.load_movielens()

    def _train_and_asset(self, X, y ,conf: deeptable.ModelConfig):
        dt = deeptable.DeepTable(config=conf)
        model, history = dt.fit(X, y, validation_split=0.2, epochs=2, batch_size=32)
        assert len(model.model.input_names) == 1

    def test_only_categorical_feature(self):
        df = self.df_bank.copy()
        X = df[['loan']]
        y = df['y']
        conf = deeptable.ModelConfig(nets=['dnn_nets'],
                                     task=consts.TASK_BINARY,
                                     metrics=['accuracy'],
                                     fixed_embedding_dim=True,
                                     embeddings_output_dim=4,
                                     apply_gbm_features=False,
                                     apply_class_weight=True,
                                     earlystopping_patience=3,)
        self._train_and_asset(X, y, conf)

    def test_only_continuous_feature(self):
        df = self.df_bank.copy()
        X = df[['duration']].astype('float32')
        y = df['y']
        conf = deeptable.ModelConfig(nets=['dnn_nets'],
                                     task=consts.TASK_BINARY,
                                     metrics=['accuracy'],
                                     fixed_embedding_dim=True,
                                     embeddings_output_dim=4,
                                     apply_gbm_features=False,
                                     apply_class_weight=True,
                                     earlystopping_patience=3,)
        self._train_and_asset(X, y, conf)

    def test_only_var_len_categorical_feature(self):
        df:pd.DataFrame = self.df_movielens.copy()
        X = df[['genres']]
        y = df['rating']
        conf = deeptable.ModelConfig(nets=['dnn_nets'],
                                     task=consts.TASK_REGRESSION,
                                     metrics=['mse'],
                                     fixed_embedding_dim=True,
                                     embeddings_output_dim=4,
                                     apply_gbm_features=False,
                                     apply_class_weight=True,
                                     earlystopping_patience=3,)
        self._train_and_asset(X, y, conf)

    def test_no_input(self):
        df:pd.DataFrame = self.df_movielens.copy()
        X = pd.DataFrame()
        y = df['rating']
        conf = deeptable.ModelConfig(nets=['dnn_nets'],
                                     task=consts.TASK_REGRESSION,
                                     metrics=['mse'],
                                     fixed_embedding_dim=True,
                                     embeddings_output_dim=4,
                                     apply_gbm_features=False,
                                     apply_class_weight=True,
                                     earlystopping_patience=3,)
        dt = deeptable.DeepTable(config=conf)
        with pytest.raises(ValueError) as err_info:
            dt.fit(X, y, validation_split=0.2, epochs=2, batch_size=32)
            print(err_info)
