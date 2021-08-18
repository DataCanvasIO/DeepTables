# -*- encoding: utf-8 -*-
from sklearn.model_selection import train_test_split

from deeptables.datasets import dsutils
import pandas as pd

from deeptables.models import deeptable


def load_iris():
    from sklearn.datasets import load_iris
    X, y = load_iris(True)
    df = pd.DataFrame(data=X, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

    def replace2str(label):
        if label == 0:
            return 'Iris-setosa'
        elif label == 1:
            return 'Iris-versicolor'
        else:
            return 'Iris-virginica'

    df['Species'] = [replace2str(l) for l in y]

    return df


df_train = load_iris()
y = df_train.pop('Species').values
X = df_train

conf = deeptable.ModelConfig(nets=["dnn_nets"],
                             metrics=['AUC'],
                             fixed_embedding_dim=True,
                             embeddings_output_dim=2,
                             apply_gbm_features=False,
                             apply_class_weight=True)
dt = deeptable.DeepTable(config=conf)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model, history = dt.fit(X_train, y_train, epochs=5)

print(dt.task)

result = dt.predict(X)
# dt.predict_proba_all()
proba = dt.predict_proba(X)

import numpy as np
print(np.unique(result))

print(dt.evaluate(X, y))




