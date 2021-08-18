# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import os
import time

from deeptables.utils import consts

homedir = f'{consts.PROJECT_NAME}_test_{time.strftime("%Y%m%d%H%M%S")}'
os.environ[consts.ENV_DEEPTABLES_HOME] = homedir
