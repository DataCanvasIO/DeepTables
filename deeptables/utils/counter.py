# -*- encoding: utf-8 -*-

_data_ = {}


def next_num(counter_name):
    _data_[counter_name] = _data_.get(counter_name, -1) + 1  # index begin from 0

    return _data_[counter_name]
