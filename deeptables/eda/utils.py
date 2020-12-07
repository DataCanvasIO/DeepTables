# -*- coding:utf-8 -*-
__author__ = 'yangjian'
"""

"""

import itertools

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


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
    from matplotlib_venn import venn2

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

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
