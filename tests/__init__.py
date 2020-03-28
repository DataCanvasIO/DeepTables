# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""
import tempfile
import os
from deeptables.utils import consts
import tests.misc

homedir = tempfile.mkdtemp()
os.environ[consts.ENV_DEEPTABLES_HOME] = homedir