# AutoML

DeepTables provide a full-pipeline AutoML library also, which completely covers the end-to-end stages of data cleaning, preprocessing, feature generation and selection, model selection and hyperparameter optimization.It is a real-AutoML tool for tabular data.

Unlike most AutoML approaches that focus on tackling the hyperparameter optimization problem of machine learning algorithms, DeepTables can put the entire process from data cleaning to algorithm selection in one search space for optimization. End-to-end pipeline optimization is more like a sequential decision process, thereby DeepTables uses reinforcement learning, Monte Carlo Tree Search, evolution algorithm combined with a meta-learner to efficiently solve such problems.  The underlying search space representation and search algorithm in DeepTables are powered by the general AutoML framework Hypernets.

## Quick start

This example demonstrate how to  train a  binary classification model with the default search spaces.

```python
from deeptables.datasets import dsutils
from deeptables.models import make_experiment
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# load data
df = dsutils.load_bank().head(10000)
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# make experiment with the default search space and run it
experiment = make_experiment(train_data.copy(), target='y', reward_metric='Precision', pos_label='yes', max_trials=5)
estimator = experiment.run()

# evaluate the estimator with classification_report
X_test = test_data.copy()
y_test = X_test.pop('y')
y_pred = estimator.predict(X_test)
print(classification_report(y_test, y_pred, digits=5))
```

## Customize the experiment

The `make_experiment` utility create a Hypernets `CompeteExperiment` instance, which intergrate many advanced features includes:
* data cleaning
* feature generation
* multicollinearity detection
* data drift detection
* feature selection
* pseudo labeling
* model ensemble
* ...

The experiment can be customized with many arguments:
```python
experiment = make_experiment(train_data, target='y',
                             cv=True, num_folds=5,
                             feature_selection=True,
                             feature_selection_strategy='quantile',
                             feature_selection_quantile=0.3,
                             reward_metric='Precision',
                             ...)

```

For more details, see [API Reference](deeptables.models.html#deeptables.models.hyper_dt.make_experiment).

## Customize the search space

DeepTables AutoML default search space defined as:

```python
def mini_dt_space():
    space = HyperSpace()
    with space.as_default():
        p_nets = MultipleChoice(
            ['dnn_nets', 'linear', 'fm_nets'], num_chosen_most=2)
        dt_module = DTModuleSpace(
            nets=p_nets,
            auto_categorize=Bool(),
            cat_remain_numeric=Bool(),
            auto_discrete=Bool(),
            apply_gbm_features=Bool(),
            gbm_feature_type=Choice([DT_consts.GBM_FEATURE_TYPE_DENSE, DT_consts.GBM_FEATURE_TYPE_EMB]),
            embeddings_output_dim=Choice([4, 10]),
            embedding_dropout=Choice([0, 0.5]),
            stacking_op=Choice([DT_consts.STACKING_OP_ADD, DT_consts.STACKING_OP_CONCAT]),
            output_use_bias=Bool(),
            apply_class_weight=Bool(),
            earlystopping_patience=Choice([1, 3, 5])
        )
        dnn = DnnModule(hidden_units=Choice([100, 200]),
                        reduce_factor=Choice([1, 0.8]),
                        dnn_dropout=Choice([0, 0.3]),
                        use_bn=Bool(),
                        dnn_layers=2,
                        activation='relu')(dt_module)
        fit = DTFit(batch_size=Choice([128, 256]))(dt_module)

    return space
```

To replace *fm_nets* with *cin_nets* and *pnn_nets*,  you can define new search space *my_dt_space* as:

```python
from deeptables.utils import consts as DT_consts
from hypernets.core.search_space import HyperSpace, Choice, Bool, MultipleChoice
from deeptables.models.hyper_dt import DTModuleSpace, DnnModule, DTFit


def my_dt_space():
    space = HyperSpace()
    with space.as_default():
        p_nets = MultipleChoice(
            ['dnn_nets', 'linear', 'cin_nets', 'pnn_nets', ], num_chosen_most=2)
        dt_module = DTModuleSpace(
            nets=p_nets,
            auto_categorize=Bool(),
            cat_remain_numeric=Bool(),
            auto_discrete=Bool(),
            apply_gbm_features=Bool(),
            gbm_feature_type=Choice([DT_consts.GBM_FEATURE_TYPE_DENSE, DT_consts.GBM_FEATURE_TYPE_EMB]),
            embeddings_output_dim=Choice([4, 10]),
            embedding_dropout=Choice([0, 0.5]),
            stacking_op=Choice([DT_consts.STACKING_OP_ADD, DT_consts.STACKING_OP_CONCAT]),
            output_use_bias=Bool(),
            apply_class_weight=Bool(),
            earlystopping_patience=Choice([1, 3, 5])
        )
        dnn = DnnModule(hidden_units=Choice([100, 200]),
                        reduce_factor=Choice([1, 0.8]),
                        dnn_dropout=Choice([0, 0.3]),
                        use_bn=Bool(),
                        dnn_layers=2,
                        activation='relu')(dt_module)
        fit = DTFit(batch_size=Choice([128, 256]))(dt_module)

    return space
```

Then, create experiment with your search space *my_dt_space*:

```python
experiment = make_experiment(train_data.copy(), target='y',
                             search_space=my_dt_space,
                             ...)
```

