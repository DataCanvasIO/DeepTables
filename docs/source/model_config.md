# ModelConfig

**ModelConfig** is the most important parameter in DT. It is used to set how to clean and preprocess the data automatically, and how to assemble various network components to building a neural nets for prediction tasks, as well as the setting of hyper-parameters of nets, etc. If you do not change any settings in ModelConfig, DT will work in most cases as well. However, you can get a better performance by tuning the parameters in ModelConfig.

We describe in detail below.

## Simple use case for ModelConfig

```python
from deeptables.models.deeptable import DeepTable, ModelConfig
from deeptables.models.deepnets import DeepFM

conf = ModelConfig(
    nets=DeepFM, # same as `nets=['linear','dnn_nets','fm_nets']`
    categorical_columns='auto', # or categorical_columns=['x1', 'x2', 'x3', ...]
    metrics=['AUC', 'accuracy'], # can be `metrics=['RootMeanSquaredError']` for regression task
    auto_categorize=True,
    auto_discrete=False,
    embeddings_output_dim=20,
    embedding_dropout=0.3,
    )
dt = DeepTable(config=conf)
dt.fit(X, y)
```

## Parameters

### nets
list of str or custom function, (default=`['dnn_nets']`)

You can use multiple components to compose neural network joint training to perform prediction tasks.

The value of nets can be any combination of component name, preset model and custom function.

**components**:

- 'dnn_nets'
- 'linear'
- 'cin_nets'
- 'fm_nets'
- 'afm_nets'
- 'opnn_nets'
- 'ipnn_nets'
- 'pnn_nets',
- 'cross_nets'
- 'cross_dnn_nets'
- 'dcn_nets',
- 'autoint_nets'
- 'fg_nets'
- 'fgcnn_cin_nets'
- 'fgcnn_fm_nets'
- 'fgcnn_ipnn_nets'
- 'fgcnn_dnn_nets'
- 'fibi_nets'
- 'fibi_dnn_nets'

**preset models**: （in package deeptables.models.deepnets）

- DeepFM 
- xDeepFM
- DCN
- PNN
- WideDeep
- AutoInt
- AFM
- FGCNN
- FibiNet

**custom function**:
```
def custom_net(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    out = layers.Dense(10)(flatten_emb_layer)
    return out
```

**examples**:
```python
from deeptables.models.deeptable import ModelConfig, DeepTable
from deeptables.models import deepnets
from tensorflow.keras import layers

#preset model
conf = ModelConfig(nets=deepnets.DeepFM)

#list of str(name of component)
conf = ModelConfig(nets=['linear','dnn_nets','cin_nets','cross_nets'])

#mixed preset model and names
conf = ModelConfig(nets=deepnets.WideDeep+['cin_nets'])

#mixed names and custom function
def custom_net(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    out = layers.Dense(10)(flatten_emb_layer)
    return out
conf = ModelConfig(nets=['linear', custom_net])
```

### categorical_columns
list of strings or 'auto', optional, (default=`'auto'`)

Only categorical features will be passed into embedding layer, and most of the components in DT are specially designed for the embedding outputs for feature extraction. Reasonable selection of categorical features is critical to model performance.


If **list of strings**, interpreted as column names.

If `'auto'`, get the categorical columns automatically. `object`, `bool` and `category` columns will be selected by default, and [auto_categorize] will no longer take effect.

If not necessary, we strongly recommend use default value `'auto'`.

### exclude_columns
list of strings, (default=[])


### pos_label
str or int, (default=`None`)

The label of positive class, used only when task is binary.


### metrics
list of strings or callable object, (default=`['accuracy']`)

List of metrics to be evaluated by the model during training and testing. Typically you will use `metrics=['accuracy']` or `metrics=['AUC']`. Every metric should be a built-in evaluation metric in tf.keras.metrics or a callable object like `r2(y_true, y_pred):...`.

**See also**: 
[https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/keras/metrics](https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf/keras/metrics
)


### auto_categorize
bool, (default=`False`)

Whether to automatically categorize eligible continuous features. 

- `True`: 
- `False`: 


### cat_exponent
float, (default=`0.5`), between 0 and 1

Only usable when auto_categrization = `True`.

Columns with `(number of unique values < number of samples ** cat_exponent)` will be treated as categorical feature.


### cat_remain_numeric
bool, (default=`True`)

Only usable when auto_categrization = `True`.

Whether continuous features transformed into categorical retain numerical features.

- `True`: 
- `False`: 

### auto_encode_label
bool, (default=`True`)

Whether to automatically perform label encoding on categorical features.


### auto_imputation
bool, (default=`True`)

Whether to automatically perform imputation on all features.

### auto_discrete
bool, (default=`False`)

Whether to discretize all continuous features into categorical features.


### fixed_embedding_dim
bool, (default=`True`)

Whether the embeddings output of all categorical features uses the same 'output_dim'. It should be noted that some components require that the output_dim of embeddings must be the same, including **FM**, **AFM**, **CIN**, **MultiheadAttention**, **SENET**, **InnerProduct**, etc.

If `False` and embedding_output_dim=0, then the output_dim of embeddings will be calculated using the following formula: 
```python
min(4 * int(pow(voc_size, 0.25)), 20)
#voc_size is the number of unique values of each feature.
```


### embeddings_output_dim
int, (default=`4`)

### embeddings_initializer
str or object, (default=`'uniform'`)

Initializer for the `embeddings` matrix.

### embeddings_regularizer
str or object, (default=`None`)

Regularizer function applied to the `embeddings` matrix.

### dense_dropout
float, (default=`0`) between 0 and 1

Fraction of the dense input units to drop.

### embedding_dropout
float, (default=`0.3`) between 0 and 1

Fraction of the embedding input units to drop.

### stacking_op
str, (default=`'add'`)

- `'add'`
- `'concat'`

### output_use_bias
bool, (default=`True`)

### optimizer
str(name of optimizer) or optimizer instance or 'auto', (default=`'auto'`)

See `tf.keras.optimizers`.

- 'auto': Automatically select optimizer based on task type.


### loss
str(name of objective function) or objective function or `tf.losses.Loss` instance or 'auto', (default='auto')

See `tf.losses`.

- 'auto': Automatically select objective function based on task type.

### home_dir
str, (default=`None`)

The home directory for saving model-related files. Each time running `fit(...)` or `fit_cross_validation(...)`, a subdirectory with a time-stamp will be created in this directory.

### monitor_metric
str, (default=`None`)

### earlystopping_patience
int, (default=`1`)

### gpu_usage_strategy
 
str, (default=`'memory_growth'`)

- `'memory_growth'`
- `'None'`

### distribute_strategy:

tensorflow.python.distribute.distribute_lib.Strategy, (default=`None`)



### dnn_params
dictionary
Only usable when 'dnn_nets' or a component using 'dnn' like 'pnn_nets','dcn_nets' included in [nets].
```
{
    'dnn_units': ((128, 0, False), (64, 0, False)),
    'dnn_activation': 'relu'}
)
```

### autoint_params
dictionary
Only usable when 'autoint_nets' included in [nets].
```
{
    'num_attention': 3,
    'num_heads': 1,
    'dropout_rate': 0,
    'use_residual': True
}
```
### fgcnn_params
dictionary
Only usable when 'fgcnn_nets' or a component using 'fgcnn' included in [nets].
```
{
    'fg_filters': (14, 16),
    'fg_widths': (7, 7),
    'fg_pool_widths': (2, 2),
    'fg_new_feat_filters': (2, 2),
}
```

### fibinet_params
dictionary
Only usable when 'fibi_nets' included in [nets].
```
{
    'senet_pooling_op': 'mean',
    'senet_reduction_ratio': 3,
    'bilinear_type': 'field_interaction',
}
```
                
### cross_params
dictionary
Only usable when 'cross_nets' included in [nets].
```
{
    'num_cross_layer': 4,
}
```

### pnn_params
dictionary
Only usable when 'pnn_nets' or 'opnn_nets' included in [nets].
```
{
    'outer_product_kernel_type': 'mat',
}
```

### afm_params
dictionary
Only usable when 'afm_nets' included in [nets].
```
{
    'attention_factor': 4,
    'dropout_rate': 0
}
```

### cin_params
dictionary
Only usable when 'cin_nets' included in [nets].
```
{
    'cross_layer_size': (128, 128),
    'activation': 'relu',
    'use_residual': False,
    'use_bias': False,
    'direct': False,
    'reduce_D': False,
}
```
