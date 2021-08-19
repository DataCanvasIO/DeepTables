# DeepTables


[![Python Versions](https://img.shields.io/pypi/pyversions/deeptables.svg)](https://pypi.org/project/deeptables)
[![TensorFlow Versions](https://img.shields.io/badge/TensorFlow-2.0+-blue.svg)](https://pypi.org/project/deeptables)
[![Downloads](https://pepy.tech/badge/deeptables)](https://pepy.tech/project/deeptables)
[![PyPI Version](https://img.shields.io/pypi/v/deeptables.svg)](https://pypi.org/project/deeptables)


[![Documentation Status](https://readthedocs.org/projects/deeptables/badge/?version=latest)](https://deeptables.readthedocs.io/)
[![Build Status](https://travis-ci.org/DataCanvasIO/deeptables.svg?branch=master)](https://travis-ci.org/DataCanvasIO/deeptables)
[![Coverage Status](https://coveralls.io/repos/github/DataCanvasIO/deeptables/badge.svg?branch=master)](https://coveralls.io/github/DataCanvasIO/deeptables?branch=master)
[![License](https://img.shields.io/github/license/DataCanvasIO/deeptables.svg)](https://github.com/DataCanvasIO/deeptables/blob/master/LICENSE)

## We Are Hiring！
Dear folks, we are opening several precious positions based in Beijing both for professionals and interns avid in AutoML/NAS, please send your resume/cv to yangjian@zetyun.com. (Application deadline: TBD.) 

## DeepTables: Deep-learning Toolkit for Tabular data
DeepTables(DT) is a easy-to-use toolkit that enables deep learning to unleash great power on tabular data.


## Overview

MLP (also known as Fully-connected neural networks) have been shown inefficient in learning distribution representation. The "add" operations of the perceptron layer have been proven poor performance to exploring multiplicative feature interactions. In most cases, manual feature engineering is necessary and this work requires extensive domain knowledge and very cumbersome. How learning feature interactions efficiently in neural networks becomes the most important problem.

Various models have been proposed to CTR prediction and continue to outperform existing state-of-the-art approaches to the late years. Well-known examples include FM, DeepFM, Wide&Deep, DCN, PNN, etc. These models can also provide good performance on tabular data under reasonable utilization.

DT aims to utilize the latest research findings to provide users with an end-to-end toolkit on tabular data.

DT has been designed with these key goals in mind:

* Easy to use, non-experts can also use.
* Provide good performance out of the box.
* Flexible architecture and easy expansion by user.

## Tutorials
Please refer to the official docs at [https://deeptables.readthedocs.io/en/latest/](https://deeptables.readthedocs.io/en/latest/).
* [Quick Start](https://deeptables.readthedocs.io/en/latest/quick_start.html)
* [Examples](https://deeptables.readthedocs.io/en/latest/examples.html)
* [ModelConfig](https://deeptables.readthedocs.io/en/latest/model_config.html)
* [Models](https://deeptables.readthedocs.io/en/latest/models.html)
* [Layers](https://deeptables.readthedocs.io/en/latest/layers.html)
* [AutoML](https://deeptables.readthedocs.io/en/latest/automl.html)

## Installation

`pip` is recommended to install DeepTables:

```bash
pip install tensorflow==2.4.2 deeptables
```

Note:
* Tensorflow is required by DeepTables, install it before running DeepTables. 
* DeepTables was tested with TensorFlow version 2.0 to 2.4, install the tested version please.

**GPU** Setup (Optional)

To use DeepTables with GPU devices, install `tensorflow-gpu` instead of `tensorflow`.

```bash
pip install tensorflow-gpu==2.4.2 deeptables
```


***Verify the installation***:

```bash
python -c "from deeptables.utils.quicktest import test; test()"
```

## Optional dependencies
Following libraries are not hard dependencies and are not automatically installed when you install DeepTables. To use all functionalities of DT, these optional dependencies must be installed.

```bash
pip install shap
```

## Example：

### A simple binary classification example
```python
import numpy as np
from deeptables.models import deeptable, deepnets
from deeptables.datasets import dsutils
from sklearn.model_selection import train_test_split

#loading data
df = dsutils.load_bank()
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

y = df_train.pop('y')
y_test = df_test.pop('y')

#training
config = deeptable.ModelConfig(nets=deepnets.DeepFM)
dt = deeptable.DeepTable(config=config)
model, history = dt.fit(df_train, y, epochs=10)

#evaluation
result = dt.evaluate(df_test,y_test, batch_size=512, verbose=0)
print(result)

#scoring
preds = dt.predict(df_test)
```

### A solution using DeepTables to win the 1st place in Kaggle Categorical Feature Encoding Challenge II

[Click here](https://github.com/DataCanvasIO/DeepTables/blob/master/examples/Kaggle%20-%20Categorical%20Feature%20Encoding%20Challenge%20II.ipynb)

## DataCanvas

![](docs/source/images/dc_logo_1.png)

DeepTables is an open source project created by [DataCanvas](https://www.datacanvas.com/). 
