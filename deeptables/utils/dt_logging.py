# -*- coding:utf-8 -*-
"""DeepTables logging module."""

import logging
from deeptables.utils import consts

_logger = logging.getLogger(consts.PROJECT_NAME)


def get_logger(logger_name=None):
    if logger_name:
        return logging.getLogger(f'{consts.PROJECT_NAME}.{logger_name}')
    else:
        return _logger
