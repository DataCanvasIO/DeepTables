# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""


def r2_c(y_true, y_pred):
    from tensorflow.keras import backend as K
    import tensorflow as tf
    y_true = tf.convert_to_tensor(y_true, dtype='float32')
    y_pred = tf.convert_to_tensor(y_pred, dtype='float32')
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    score = (1 - SS_res / (SS_tot + K.epsilon()))
    return score
