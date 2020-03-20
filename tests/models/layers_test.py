# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from deeptables.models.layers import MultiColumnEmbedding
from tensorflow.keras import layers, models
import numpy as np


class Test_Layers:

    def test_multicolumns_embedding(self):
        train_x = np.random.randint(1, 1000, (256, 20))
        y = np.random.uniform(0., 1., 256)

        input_dims = list(np.max(train_x, axis=0) + 1)
        output_dims = list(np.random.randint(4, 10, 20))  # [10 for i in range(20)]
        input = layers.Input((20,))
        mce = MultiColumnEmbedding(input_dims, output_dims, dropout_rate=0.5)
        embeddings = mce(input)
        x = layers.Flatten()(layers.concatenate(embeddings))
        x = layers.Dense(10)(x)
        out = layers.Dense(1)(x)
        model = models.Model(inputs=input, outputs=out)
        model.compile(loss='mse')
        model.fit(train_x, y)
        assert len(embeddings), 20
        assert embeddings[0].shape, (None, 1, 10)
