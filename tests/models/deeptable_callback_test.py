# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from deeptables.models import deeptable
from deeptables.datasets import dsutils
import os
import tempfile


class Test_DeepTable_Callback:
    def test_callback_injection(self):
        print("Loading datasets...")
        df_train = dsutils.load_adult()
        self.y = df_train.pop(14).values
        self.X = df_train
        path = tempfile.mkdtemp()
        conf = deeptable.ModelConfig(metrics=['AUC'],
                                     apply_gbm_features=False,
                                     auto_discrete=False,
                                     home_dir=path,
                                     )

        self.dt = deeptable.DeepTable(config=conf)

        mcp = ModelCheckpoint(path,
                              'val_auc',
                              verbose=0,
                              save_best_only=False,
                              save_weights_only=False,
                              mode='max',
                              save_freq='epoch',
                              )
        callbacks = [mcp]
        self.X_train, \
        self.X_test, \
        self.y_train, \
        self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.model, self.history = self.dt.fit(self.X_train, self.y_train, epochs=1, callbacks=callbacks)

        files = os.listdir(path)
        assert 'saved_model.pb' in files
