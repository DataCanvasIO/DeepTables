# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import numpy as np
import pandas as pd
from eli5.permutation_importance import get_score_importances as eli5_importances

from hypernets.tabular import get_tool_box


def get_score_importances(dt_model, X, y, metric, n_iter=5, mode='min'):
    columns = X.columns.to_list()
    metric = metric.lower()

    def score(X_s, y_s) -> float:
        df = pd.DataFrame(X_s)
        df.columns = columns
        if metric in ['auc', 'log_loss']:
            y_proba = dt_model.predict_proba(df)
            y_pred = y_proba
        else:
            y_pred = dt_model.predict(df)
            y_proba = y_pred
        del df
        dict = get_tool_box(y_s).metrics.calc_score(y_s, y_pred, y_proba, [metric], dt_model.task, dt_model.pos_label)
        print(f'score:{dict}')
        if mode == 'min':
            return -dict[metric]
        elif mode == 'max':
            return dict[metric]
        else:
            raise ValueError(f'Unsupported mode:{mode}')

    base_score, score_decreases = eli5_importances(score, X.values, y, n_iter=n_iter)
    feature_importances = np.stack([columns, np.mean(score_decreases, axis=0)], axis=1)
    feature_importances = np.array(sorted(feature_importances, key=lambda fi: fi[1], reverse=True))
    return feature_importances


def select_features(feature_importances, threshold=0.):
    selected_columns = [fi[0] for fi in feature_importances if float(fi[1]) > threshold]
    discard_columns = [fi[0] for fi in feature_importances if float(fi[1]) <= threshold]
    return selected_columns, discard_columns
