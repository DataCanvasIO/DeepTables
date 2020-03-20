# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import os
from deeptables.utils import consts
import tempfile

homedir = tempfile.mkdtemp()
os.environ[consts.ENV_DEEPTABLES_HOME] = homedir