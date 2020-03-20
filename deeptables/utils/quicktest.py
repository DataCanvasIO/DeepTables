# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import numpy as np
import pandas as pd
from deeptables.models import deepnets, deeptable


def test():
    X = pd.DataFrame(np.random.random((100, 4)))
    y = pd.Series(np.random.randint(0, 2, 100))
    dt = deeptable.DeepTable(deeptable.ModelConfig(nets=deepnets.DeepFM))
    dt.fit(X, y)
