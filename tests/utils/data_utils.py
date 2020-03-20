# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
from deeptables.datasets import dsutils


class Test_data_utils:
    def test_load_data(self):
        df_adult = dsutils.load_adult()
        df_glass = dsutils.load_glass_uci()
        df_hd = dsutils.load_heart_disease_uci()
        df_bank = dsutils.load_bank()
        assert df_adult.shape, (32561, 15)
        assert df_glass.shape, (214, 11)
        assert df_hd.shape, (303, 14)
        assert df_bank.shape, (108504, 18)
