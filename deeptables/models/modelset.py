# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
from ..utils import consts


class ModelInfo:
    def __init__(self, type, name, model, score, **meta):
        self.type = type
        self.name = name
        self.model = model
        self.score = self.dict_lower_keys(score)
        self.meta = meta

        if len(score) <= 0 and meta['history'] is not None:
            history = meta['history']
            self.score = {k.lower(): history[k][-1] for k in history.keys()}

    def dict_lower_keys(self, dict):
        if dict is None:
            return {}
        ldict = {}
        for k, v in dict.items():
            ldict[k.lower()] = v
        return ldict

    def get_score(self, metric_name):
        score = self.score.get(metric_name.lower())
        if score is None:
            return 0
        else:
            return score


class ModelSet:
    def __init__(self, metric=consts.METRIC_NAME_AUC, best_mode=consts.MODEL_SELECT_MODE_MAX):
        self.best_mode = best_mode
        self.metric = metric.lower()
        self.__models = []

    def clear(self):
        self.__models = []

    def push(self, modelinfo):
        if self.get_modelinfo(modelinfo.name) is not None:
            raise ValueError(f'Duplicate model name is not allowed，model named "{modelinfo.name}" already exists。')
        self.__models.append(modelinfo)

    def get_modelinfo(self, name):
        for mi in self.__models:
            if mi.name == name:
                return mi
        return None

    def best_model(self):
        if len(self.__models) <= 0:
            raise ValueError('Model set is empty.')
        self.__sort()
        return self.__models[0]

    def get_models(self, type=None):
        if type is not None:
            return [m.model for m in self.__models if m.type == type]
        else:
            return [m.model for m in self.__models]

    def get_modelinfos(self, type=None):
        if type is not None:
            return [m for m in self.__models if m.type == type]
        else:
            return [m for m in self.__models]

    def top_n(self, top=0, type=None):
        self.__sort()
        models = self.get_modelinfos(type=type)
        if top <= 0:
            top = len(models)
        if len(models) >= top:
            models = models[0:top]
        return models

    def leaderboard(self, top=0, type=None):
        models = self.top_n(top, type=type)
        rows = []
        for m in models:
            df = pd.DataFrame(np.array(list(m.score.values())).reshape(1, -1))
            keys = list(m.score.keys())
            try:
                index = keys.index(self.metric)
                keys[index] = '*' + self.metric
            except:
                print(f'Not found sort-metric:{self.metric} in metrics:{keys}.')
            df.columns = keys
            df.insert(0, 'type', [m.type])
            df.insert(0, 'model', [m.name])
            rows.append(df)
        if len(rows) <= 0:
            return None
        board = pd.concat(rows, axis=0).reset_index()
        board.drop(['index'], axis=1, inplace=True)
        board.insert(0, 'model', board.pop('model'))
        return board

    def __sort(self):
        best_mode = self.best_mode
        if best_mode == consts.MODEL_SELECT_MODE_AUTO:
            if self.metric.lower() in ['acc', 'accuracy', 'auc', 'recall', 'precision']:
                best_mode = consts.MODEL_SELECT_MODE_MAX
            else:
                best_mode = consts.MODEL_SELECT_MODE_MIN
        reverse = best_mode == consts.MODEL_SELECT_MODE_MAX
        self.__models.sort(key=lambda x: x.get_score(self.metric), reverse=reverse)
