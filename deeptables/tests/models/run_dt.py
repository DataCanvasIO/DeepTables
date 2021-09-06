import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from deeptables.datasets import dsutils
from deeptables.models import deeptable, deepnets


def run(distribute_strategy=None, batch_size=32, epochs=5):
    # loading data
    df = dsutils.load_bank()
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    y = df_train.pop('y')
    y_test = df_test.pop('y')

    # training
    config = deeptable.ModelConfig(nets=deepnets.DeepFM, earlystopping_patience=999, apply_class_weight=True,
                                   distribute_strategy=distribute_strategy, )
    dt = deeptable.DeepTable(config=config)
    model, history = dt.fit(df_train, y, batch_size=batch_size, epochs=epochs)

    # evaluation
    result = dt.evaluate(df_test, y_test, verbose=0)
    print('score:', result)

    # scoring
    preds = dt.predict(df_test)
    uniques = np.unique(preds, return_counts=True)
    print({k: v for k, v in zip(*uniques)})


if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    bs = int(os.environ.get('BATCH_SIZE', '32'))
    es = int(os.environ.get('EPOCHS', '5'))

    if len(gpus) < 2:
        run(batch_size=bs, epochs=es)
    else:
        strategy = tf.distribute.MirroredStrategy()
        # with strategy.scope():
        run(distribute_strategy=strategy, batch_size=len(gpus) * bs, epochs=es)

    print('done')
