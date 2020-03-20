# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

from sklearn.model_selection import train_test_split

from deeptables.models import deeptable

from deeptables.datasets import dsutils


class Test_DeepTable:

    def test_embeddings_output_dim(self):
        print("Loading datasets...")
        df_train = dsutils.load_adult().head(1000)
        y = df_train.pop(14).values
        X = df_train

        conf = deeptable.ModelConfig(fixed_embedding_dim=False, embeddings_output_dim=0)
        dt = deeptable.DeepTable(config=conf)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model, history = dt.fit(X_train, y_train, epochs=1)
