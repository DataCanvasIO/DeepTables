# Quick-Start

## Installation Guide

### Requirements
**Python 3**: DT requires Python version 3.6 or 3.7. 

**Tensorflow** >= 2.0.0: DT is based on TensorFlow. Please follow this [tutorial](https://www.tensorflow.org/install/pip) to install TensorFlow for python3.


### Install DeepTables

```shell script
pip install deeptables
```

**GPU** Setup (Optional): If you have GPUs on your machine and want to use them to accelerate the training, you can use the following command.
```shell script
pip install deeptables[gpu]
```
                     


## Getting started: 5 lines to DT

### Supported Tasks
DT can be use to solve **classification** and **regression** prediction problems on tabular data.

### Simple Exapmle
DT supports these tasks with extremely simple interface without dealing with data cleaning and feature engineering. You don't even specify the task type, DT will automatically infer.
```
from deeptables.models.deeptable import DeepTable, ModelConfig
from deeptables.models.deepnets import DeepFM

dt = DeepTable(ModelConfig(nets=DeepFM))
dt.fit(X, y)
preds = dt.predict(X_test)
```
