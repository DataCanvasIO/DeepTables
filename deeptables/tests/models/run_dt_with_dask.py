import os

import tensorflow as tf

from deeptables.datasets import dsutils
from deeptables.models import deeptable, deepnets
from hypernets.tabular import dask_ex as dex
from hypernets.tabular.metrics import calc_score
from hypernets.tests.tabular.dask_transofromer_test import setup_dask


def run(distribute_strategy=None, batch_size=32, epochs=5):
    # loading data
    df = dsutils.load_bank_by_dask()
    df_train, df_test = dex.train_test_split(df, test_size=0.2, random_state=42)

    y = df_train.pop('y')
    y_test = df_test.pop('y')

    # training
    config = deeptable.ModelConfig(nets=deepnets.DeepFM, earlystopping_patience=5,
                                   distribute_strategy=distribute_strategy, )
    dt = deeptable.DeepTable(config=config)
    model, history = dt.fit(df_train, y, batch_size=batch_size, epochs=epochs)

    # save
    model_path = 'model_by_dask'
    dt.save(model_path)
    print(f'saved to {model_path}')

    # evaluation
    model_path = 'model_by_dask'
    dt2 = deeptable.DeepTable.load(model_path)
    result = dt2.evaluate(df_test, y_test, batch_size=512, verbose=0)
    print('score:', result)

    # scoring
    preds = dt2.predict(df_test, batch_size=512, )
    proba = dt2.predict_proba(df_test, batch_size=512, )
    print(calc_score(y_test, preds, proba, metrics=['accuracy', 'auc']))


if __name__ == '__main__':
    setup_dask(None)
    gpus = tf.config.list_physical_devices('GPU')
    bs = int(os.environ.get('BATCH_SIZE', '32'))
    es = int(os.environ.get('EPOCHS', '5'))

    if len(gpus) < 2:
        run(batch_size=bs, epochs=es)
    else:
        strategy = tf.distribute.MirroredStrategy()
        run(distribute_strategy=strategy, batch_size=len(gpus) * bs, epochs=es)

    print('done')
