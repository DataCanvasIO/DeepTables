# -*- coding:utf-8 -*-
__author__ = 'yangjian'
import tensorflow as tf
from keras.src.ops import dtype

"""

"""
#
def cast(var, dtype):
    if isinstance(var, tf.Tensor):
        return tf.cast(var, dtype=dtype)
    else:
        return tf.convert_to_tensor(var, dtype=dtype)

def r2_c(y_true, y_pred):
    from keras import backend as K

    y_true = cast(y_true, 'float32')
    y_pred = cast(y_pred, 'float32')

    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    score = (1 - SS_res / (SS_tot + K.epsilon()))
    return score
