# -*- coding:utf-8 -*-
"""

"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.model_selection import train_test_split

from deeptables.datasets import dsutils
from deeptables.models import make_experiment
# from hypergbm import make_experiment
from hypernets.tabular import get_tool_box


def train(data, target, reward_metric):
    df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)
    y_test = df_test.pop(target)

    experiment = make_experiment(df_train, target=target, test_data=df_test,
                                 # reward_metric='AUC',
                                 reward_metric=reward_metric,
                                 random_state=345,
                                 log_level='info',
                                 )
    estimator = experiment.run(max_trials=3, batch_size=128, epochs=1, verbose=0, )

    y_pred = estimator.predict(df_test)
    calc_score = get_tool_box(y_test).metrics.calc_score
    if experiment.task == 'regression':
        result = calc_score(y_test, y_pred, None, task=experiment.task, metrics=[reward_metric, 'rmse', ])
    else:
        y_proba = estimator.predict_proba(df_test)
        result = calc_score(y_test, y_pred, y_proba, task=experiment.task,
                            metrics=[reward_metric, 'auc', 'accuracy', 'f1', 'recall', 'precision'])
    # result = calc_score(y_test, y_pred, y_proba, metrics=['auc', 'f1', 'recall']))
    print(result)


def run_boston():
    train(dsutils.load_boston(), 'target', 'RootMeanSquaredError')


def run_blood():
    train(dsutils.load_blood(), 'Class', 'AUC')


def run_glass_uci():
    df = dsutils.load_glass_uci()
    df.columns = [f'col_{c}' if c != 10 else 'y' for c in df.columns.to_list()]
    # train(df, 'y', 'Recall')
    train(df, 'y', 'AUC')


if __name__ == '__main__':
    # run_glass_uci()
    # run_blood()
    run_boston()
