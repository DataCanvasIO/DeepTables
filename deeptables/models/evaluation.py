# -*- coding:utf-8 -*-
__author__ = 'yangjian'

import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_log_error, accuracy_score, \
    mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score, f1_score
from ..utils import consts


def calc_score(y_true, y_proba, y_preds, metrics, task, pos_label=1):
    score = {}
    if len(y_proba.shape) == 2 and y_proba.shape[-1] == 1:
        y_proba = y_proba.reshape(-1)
    if len(y_preds.shape) == 2 and y_preds.shape[-1] == 1:
        y_preds = y_preds.reshape(-1)
    for metric in metrics:
        if callable(metric):
            score[metric.__name__] = metric(y_true, y_preds)
        else:
            metric = metric.lower()
            if task == consts.TASK_MULTICLASS:
                average = 'micro'
            else:
                average = 'binary'

            if metric == 'auc':
                if len(y_proba.shape) == 2:
                    score['auc'] = roc_auc_score(y_true, y_proba, multi_class='ovo')
                else:
                    score['auc'] = roc_auc_score(y_true, y_proba)

            elif metric == 'accuracy':
                if y_proba is None:
                    score['accuracy'] = 0
                else:
                    score['accuracy'] = accuracy_score(y_true, y_preds)
            elif metric == 'recall':
                score['recall'] = recall_score(y_true, y_preds, average=average, pos_label=pos_label)
            elif metric == 'precision':
                score['precision'] = precision_score(y_true, y_preds, average=average, pos_label=pos_label)
            elif metric == 'f1':
                score['f1'] = f1_score(y_true, y_preds, average=average, pos_label=pos_label)

            elif metric == 'mse':
                score['mse'] = mean_squared_error(y_true, y_preds)
            elif metric == 'mae':
                score['mae'] = mean_absolute_error(y_true, y_preds)
            elif metric == 'msle':
                score['msle'] = mean_squared_log_error(y_true, y_preds)
            elif metric == 'rmse':
                score['rmse'] = np.sqrt(mean_squared_error(y_true, y_preds))
            elif metric == 'rootmeansquarederror':
                score['rootmeansquarederror'] = np.sqrt(mean_squared_error(y_true, y_preds))
            elif metric == 'r2':
                score['r2'] = r2_score(y_true, y_preds)

    return score
