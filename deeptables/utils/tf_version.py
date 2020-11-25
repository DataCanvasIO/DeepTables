# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import tensorflow as tf
from packaging.version import parse


def tf_less_than(version):
    return parse(tf.__version__) < parse(version)


def tf_greater_than(version):
    return parse(tf.__version__) > parse(version)
