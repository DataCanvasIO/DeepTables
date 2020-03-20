# -*- coding:utf-8 -*-

import pandas as pd
from category_encoders import TargetEncoder
from sklearn.model_selection import StratifiedKFold

from ..utils import dt_logging

logger = dt_logging.get_logger(__name__)


def target_encoding(train, target, test=None, feat_to_encode=None, smooth=0.2, random_state=9527):
    print('Target encoding...')
    train.sort_index(inplace=True)
    target = train.pop(target)
    if feat_to_encode is None:
        feat_to_encode = train.columns.tolist()
    smoothing = smooth
    oof = pd.DataFrame([])
    for tr_idx, oof_idx in StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True).split(train, target):
        ce_target_encoder = TargetEncoder(cols=feat_to_encode, smoothing=smoothing)
        ce_target_encoder.fit(train.iloc[tr_idx, :], target.iloc[tr_idx])
        oof = oof.append(ce_target_encoder.transform(train.iloc[oof_idx, :]), ignore_index=False)
    ce_target_encoder = TargetEncoder(cols=feat_to_encode, smoothing=smoothing)
    ce_target_encoder.fit(train, target)
    train = oof.sort_index()
    if test is not None:
        test = ce_target_encoder.transform(test)
    features = list(train)
    print('Target encoding done!')
    return train, test, features, target


def target_rate_encodeing(feat_to_encode, target, df, mode='order'):  # mode:order/rate
    new_df = None
    for col in feat_to_encode:
        df[col] = df[col].astype('str').fillna('-1')
        data = df[[col, target]].groupby(col)[target].value_counts().unstack()
        data['rate'] = data[1] / (data[0] + data[1])
        data.sort_values(by=['rate'], inplace=True)
        # display(datasets.style.highlight_max(color='lightgreen').highlight_min(color='#cd4f39'))
        data = data.reset_index()
        if mode == 'order':
            dict_ord = {k: i + 1 for i, k in enumerate(data[col].values)}
            nc = df[col].map(dict_ord).astype('int32')
        else:
            dict_ord = {k[0]: k[1] for k in data[[col, 'rate']].values}
            nc = df[col].map(dict_ord)
        nn = f'{col}_tre'
        if new_df is None:
            new_df = pd.DataFrame(nc, columns=[nn])
        else:
            new_df[nn] = nc
    return df
