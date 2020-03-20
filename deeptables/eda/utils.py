# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import numpy as np
import pandas as pd
import itertools
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib_venn import venn2



def columns_info(dataframe, topN=10):
    #if not isinstance(self, pd.DataFrame):
    #    raise TypeError('object must be an instance of pandas DataFrame')
    #dataframe = self
    max_row = dataframe.shape[0]
    print(f'Shape: {dataframe.shape}')

    info = dataframe.dtypes.to_frame()
    info.columns = ['DataType']
    info['#Nulls'] = dataframe.isnull().sum()
    info['#Uniques'] = dataframe.nunique()

    # stats
    info['Min'] = dataframe.min(numeric_only=True)
    info['Mean'] = dataframe.mean(numeric_only=True)
    info['Max'] = dataframe.max(numeric_only=True)
    info['Std'] = dataframe.std(numeric_only=True)

    # top 10 values
    info[f'top{topN} val'] = 0
    info[f'top{topN} cnt'] = 0
    info[f'top{topN} raito'] = 0
    for c in info.index:
        vc = dataframe[c].value_counts().head(topN)
        val = list(vc.index)
        cnt = list(vc.values)
        raito = list((vc.values / max_row).round(2))
        info.loc[c, f'top{topN} val'] = str(val)
        info.loc[c, f'top{topN} cnt'] = str(cnt)
        info.loc[c, f'top{topN} raito'] = str(raito)
    return info


def top_categories(df, category_feature, topN=30):
    return df[category_feature].value_counts().head(topN).index


def count_categories(df, category_features, topN=30, sort='freq', df2=None):
    for c in category_features:
        target_value = df[c].value_counts().head(topN).index
        if sort == 'freq':
            order = target_value
        elif sort == 'alphabetic':
            order = df[c].value_counts().head(topN).sort_index().index

        if df2 is not None:
            plt.subplot(1, 2, 1)
        sns.countplot(x=c, data=df[df[c].isin(order)], order=order)
        plt.xticks(rotation=90)

        if df2 is not None:
            plt.subplot(1, 2, 2)
            sns.countplot(x=c, data=df2[df2[c].isin(order)], order=order)
            plt.xticks(rotation=90)

        if df2 is not None:
            plt.suptitle(f'{c} TOP{topN}', size=25)
        else:
            plt.title(f'{c} TOP{topN}', size=25)
        plt.tight_layout()
        plt.show()

    return


def hist_continuous(df, continuous_features, bins=30, df2=None):
    for c in continuous_features:
        if df2 is not None:
            plt.subplot(1, 2, 1)
        df[c].hist(bins=bins)

        if df2 is not None:
            plt.subplot(1, 2, 2)
            df2[c].hist(bins=bins)

        if df2 is not None:
            plt.suptitle(f'{c}', size=25)
        else:
            plt.title(f'{c}', size=25)
        plt.tight_layout()
        plt.show()

    return


def venn_diagram(train, test, category_features, names=('train', 'test'), figsize=(18, 13)):
    """
    category_features: max==6
    """
    n = int(np.ceil(len(category_features) / 2))
    plt.figure(figsize=figsize)

    for i, c in enumerate(category_features):
        plt.subplot(int(f'{n}2{i + 1}'))
        venn2([set(train[c].unique()), set(test[c].unique())],
              set_labels=names)
        plt.title(f'{c}', fontsize=18)
    plt.show()

    return


def split_seq(iterable, size):
    """
    In: list(split_seq(range(9), 4))
    Out: [[0, 1, 2, 3], [4, 5, 6, 7], [8]]
    """
    it = iter(iterable)
    item = list(itertools.islice(it, size))
    while item:
        yield item
        item = list(itertools.islice(it, size))

