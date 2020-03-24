# Exapmles

## Binary Classification

This example demonstrate how to use WideDeep nets to solve a binary classification prediction problem. 

```python
from deeptables.models.deeptable import DeepTable, ModelConfig
from deeptables.models.deepnets import WideDeep
from deeptables.datasets import dsutils
from sklearn.model_selection import train_test_split

#Adult Data Set from UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Adult
df_train = dsutils.load_adult()
y = df_train.pop(14)
X = df_train

#`auto_discrete` is used to decide wether to discretize continous varibles automatically.
conf = ModelConfig(nets=WideDeep, metrics=['AUC','accuracy'], auto_discrete=True)
dt = DeepTable(config=conf)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model, history = dt.fit(X_train, y_train, epochs=100)

score = dt.evaluate(X_test, y_test)

preds = dt.predict(X_test)
```

## Multiclass Classification

This simple example demonstrate how to use a DNN(MLP) nets to solve a multiclass task on MNIST dataset.

```python
from deeptables.models import deeptable
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

conf = deeptable.ModelConfig(nets=['dnn_nets'], optimizer=keras.optimizers.RMSprop())
dt = deeptable.DeepTable(config=conf)

model, history = dt.fit(x_train, y_train, epochs=10)

score = dt.evaluate(x_test, y_test, batch_size=512, verbose=0)

preds = dt.predict(x_test)
```

## Regression
This example shows how to use DT to predicting Boston housing price.

```python
from deeptables.models.deeptable import DeepTable, ModelConfig
from deeptables.datasets import dsutils
from sklearn.model_selection import train_test_split

df_train = dsutils.load_boston()
y = df_train.pop('target')
X = df_train

conf = ModelConfig(
    metrics=['RootMeanSquaredError'], 
    nets=['dnn_nets'],
    dnn_params={
        'dnn_units': ((256, 0.3, True), (256, 0.3, True)),
        'dnn_activation': 'relu',
    },
    earlystopping_patience=5,
)

dt = DeepTable(config=conf)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model, history = dt.fit(X_train, y_train, epochs=100)

score = dt.evaluate(X_test, y_test)
```