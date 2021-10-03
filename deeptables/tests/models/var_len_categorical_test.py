# -*- encoding: utf-8 -*-
from deeptables.datasets import dsutils
from deeptables.models import deeptable
from deeptables.utils import consts
from hypernets.tabular import get_tool_box


class TestVarLenCategoricalFeature:

    def setup_class(cls):
        cls.df = dsutils.load_movielens().drop(['timestamp', "title"], axis=1)

    def test_var_categorical_feature(self):
        X = self.df.copy()
        y = X.pop('rating').values.astype('float32')

        conf = deeptable.ModelConfig(nets=['dnn_nets'],
                                     task=consts.TASK_REGRESSION,
                                     categorical_columns=["movie_id", "user_id", "gender", "occupation", "zip", "title",
                                                          "age"],
                                     metrics=['mse'],
                                     fixed_embedding_dim=True,
                                     embeddings_output_dim=4,
                                     apply_gbm_features=False,
                                     apply_class_weight=True,
                                     earlystopping_patience=5,
                                     var_len_categorical_columns=[('genres', "|", "max")])

        dt = deeptable.DeepTable(config=conf)

        X_train, X_validation, y_train, y_validation = get_tool_box(X).train_test_split(X, y, test_size=0.2)

        model, history = dt.fit(X_train, y_train, validation_data=(X_validation, y_validation),
                                epochs=10, batch_size=32)

        assert 'genres' in model.model.input_names
