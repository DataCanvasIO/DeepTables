from deeptables.models.hyper_dt import tiny_dt_space, make_experiment
from hypernets.tabular import dask_ex as dex
from hypernets.tabular.datasets import dsutils
from hypernets.tabular.metrics import calc_score
from hypernets.tests.tabular.dask_transofromer_test import setup_dask


def run_compete_experiment_with_heart_disease(init_kwargs, run_kwargs, with_dask=False):
    df = dsutils.load_heart_disease_uci()
    target = 'target'

    if with_dask:
        setup_dask(None)
        df = dex.dd.from_pandas(df, npartitions=1)

    train_data, test_data = dex.train_test_split(df, test_size=0.2, random_state=7)
    train_data, eval_data = dex.train_test_split(train_data, test_size=0.3, random_state=7)
    y_test = test_data.pop(target)

    init_kwargs = {
        'searcher': 'random',
        'search_space': tiny_dt_space,
        'reward_metric': 'AUC',
        'ensemble_size': 0,
        'drift_detection': False,
        **init_kwargs
    }
    run_kwargs = {
        'max_trials': 3,
        'batch_size': 128,
        'epochs': 1,
        **run_kwargs
    }
    experiment = make_experiment(train_data, target='target', eval_data=eval_data, test_data=test_data, **init_kwargs)
    estimator = experiment.run(**run_kwargs)

    assert estimator is not None

    preds = estimator.predict(test_data)
    proba = estimator.predict_proba(test_data)

    score = calc_score(y_test, preds, proba, metrics=['AUC', 'accuracy', 'f1', 'recall', 'precision'])
    print('evaluate score:', score)
    assert score


def test_experiment_basis():
    run_compete_experiment_with_heart_disease({}, {})


def test_experiment_basis_dask():
    run_compete_experiment_with_heart_disease({}, {}, with_dask=True)


# def test_with_jobs():
#     run_compete_experiment_with_bank_data(dict(cv=True), dict(n_jobs=3))


def test_experiment_without_cv():
    run_compete_experiment_with_heart_disease(dict(cv=False), {})


def test_experiment_without_cv_dask():
    run_compete_experiment_with_heart_disease(dict(cv=False), {}, with_dask=True)


def test_experiment_with_ensemble():
    run_compete_experiment_with_heart_disease(dict(ensemble_size=3, cv=False), {})


def test_experiment_with_ensemble_dask():
    run_compete_experiment_with_heart_disease(dict(ensemble_size=3, cv=False), {}, with_dask=True)


def test_experiment_with_cv_ensemble():
    run_compete_experiment_with_heart_disease(dict(ensemble_size=3, cv=True), {})


def test_experiment_with_cv_ensemble_dask():
    run_compete_experiment_with_heart_disease(dict(ensemble_size=3, cv=True), {}, with_dask=True)
