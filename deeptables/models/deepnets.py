# -*- coding:utf-8 -*-
import six
from inspect import signature
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Flatten, BatchNormalization, Activation, Dropout
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object

from . import layers

WideDeep = ['linear', 'dnn_nets']
DeepFM = ['linear', 'fm_nets', 'dnn_nets']
xDeepFM = ['linear', 'cin_nets', 'dnn_nets']
AutoInt = ['autoint_nets']
DCN = ['dcn_nets']
FGCNN = ['fgcnn_dnn_nets']
FiBiNet = ['fibi_dnn_nets']
PNN = ['pnn_nets']
AFM = ['afm_nets']


# nets = ['linear', 'cin_nets', 'fm_nets', 'afm_nets', 'opnn_nets', 'ipnn_nets', 'pnn_nets',
#         'dnn2_nets', 'dnn_nets', 'cross_nets', 'widecross_nets', 'cross_dnn_nets', 'dcn_nets',
#         'autoint_nets', 'fg_nets', 'fgcnn_cin_nets', 'fgcnn_fm_nets', 'fgcnn_ipnn_nets',
#         'fgcnn_dnn_nets', 'fibi_nets', 'fibi_dnn_nets']


def linear(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    Linear(order-1) interactions
    """
    x = None
    x_emb = None
    if embeddings is not None and len(embeddings) > 0:
        concat_embeddings = Concatenate(axis=1, name='concat_linear_embedding')(embeddings)
        x_emb = tf.reduce_sum(concat_embeddings, axis=-1, name='linear_reduce_sum')

    if x_emb is not None and dense_layer is not None:
        x = Concatenate(name='concat_linear_emb_dense')([x_emb, dense_layer])
        # x = BatchNormalization(name='bn_linear_emb_dense')(x)
    elif x_emb is not None:
        x = x_emb
    elif dense_layer is not None:
        x = dense_layer
    else:
        raise ValueError('No input layer exists.')
    input_shape = x.shape
    x = Dense(1, activation=None, use_bias=False, name='linear_logit')(x)
    model_desc.add_net('linear', input_shape, x.shape)
    return x


def cin_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """ Compressed Interaction Network (CIN), with the following considerations: (1) interactions
    are applied at vector-wise level, not at bit-wise level; (2) high-order feature interactions
    is measured explicitly; (3) the complexity of network will not grow exponentially with the degree
    of interactions.
    """
    if embeddings is None or len(embeddings) <= 0:
        model_desc.add_net('cin', (None), (None))
        return None
    cin_concat = Concatenate(axis=1, name='concat_cin_embedding')(embeddings)
    cin_output = layers.CIN(params=config.cin_params)(cin_concat)
    model_desc.add_net('cin', cin_concat.shape, cin_output.shape)
    return cin_output


def fm_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    FM models pairwise(order-2) feature interactions
    """
    if embeddings is None or len(embeddings) <= 0:
        model_desc.add_net('fm', (None), (None))
        return None
    fm_concat = Concatenate(axis=1, name='concat_fm_embedding')(embeddings)
    fm_output = layers.FM(name='fm_layer')(fm_concat)
    model_desc.add_net('fm', fm_concat.shape, fm_output.shape)
    return fm_output


def afm_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """Attentional Factorization Machine (AFM), which learns the importance of each feature interaction
    from datasets via a neural attention network.
    """
    if embeddings is None or len(embeddings) <= 0:
        return None
    afm_output = layers.AFM(params=config.afm_params, name='afm_layer')(embeddings)
    model_desc.add_net('afm', f'list({len(embeddings)})', afm_output.shape)
    return afm_output


def opnn_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    Outer Product-based Neural Network
    OuterProduct+DNN
    """
    op = layers.OuterProduct(config.pnn_params, name='outer_product_layer')(embeddings)
    model_desc.add_net('opnn-outer_product', f'list({len(embeddings)})', op.shape)

    concat_all = Concatenate(name='concat_opnn_all')([op, concat_emb_dense])
    x_dnn = dnn(concat_all, config.dnn_params, cellname='opnn')
    model_desc.add_net('opnn-dnn', concat_all.shape, x_dnn.shape)
    return x_dnn


def ipnn_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    Inner Product-based Neural Network
    InnerProduct+DNN
    """

    ip = layers.InnerProduct(name='inner_product_layer')(embeddings)
    model_desc.add_net('ipnn-inner_product', f'list({len(embeddings)})', ip.shape)

    concat_all = Concatenate(name='concat_ipnn_all')([ip, concat_emb_dense])
    x_dnn = dnn(concat_all, config.dnn_params, cellname='ipnn')
    model_desc.add_net('ipnn-dnn', concat_all.shape, x_dnn.shape)
    return x_dnn


def pnn_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    Concatenation of inner product and outer product + DNN
    """

    ip = layers.InnerProduct(name='pnn_inner_product_layer')(embeddings)
    model_desc.add_net('pnn-inner_product', f'list({len(embeddings)})', ip.shape)

    op = layers.OuterProduct(params=config.pnn_params, name='pnn_outer_product_layer')(embeddings)
    model_desc.add_net('pnn-outer_product', f'list({len(embeddings)})', op.shape)

    concat_all = Concatenate(name='concat_pnn_all')([ip, op, concat_emb_dense])
    x_dnn = dnn(concat_all, config.dnn_params, cellname='pnn')
    model_desc.add_net('pnn-dnn', concat_all.shape, x_dnn.shape)
    return x_dnn


def dnn_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    MLP (fully-connected feed-forward neural nets)
    """
    x_dnn = dnn(concat_emb_dense, config.dnn_params)
    model_desc.add_net('dnn', concat_emb_dense.shape, x_dnn.shape)
    return x_dnn


def cross_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    The Cross networks is composed of cross layers to apply explicit feature crossing in an efficient way.
    """
    cross = layers.Cross(params=config.cross_params, name='cross_layer')(concat_emb_dense)
    model_desc.add_net('cross', concat_emb_dense.shape, cross.shape)
    return cross


def cross_dnn_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    Cross nets -> DNN -> logit_out
    """
    x = concat_emb_dense
    cross = layers.Cross(params=config.cross_params, name='cross_dnn_layer')(x)
    model_desc.add_net('cross_dnn-cross', x.shape, cross.shape)

    x_dnn = dnn(cross, config.dnn_params, cellname='cross_dnn')
    model_desc.add_net('cross_dnn-dnn', cross.shape, x_dnn.shape)
    return x_dnn


def dcn_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    Concat the outputs from Cross nets and DNN nets and feed into a standard logits layer
    """
    x = concat_emb_dense
    cross_out = layers.Cross(params=config.cross_params, name='dcn_cross_layer')(x)
    model_desc.add_net('dcn-widecross', x.shape, cross_out.shape)

    dnn_out = dnn(x, config.dnn_params, cellname='dcn')
    model_desc.add_net('dcn-dnn2', x.shape, dnn_out.shape)

    stack_out = Concatenate(name='concat_cross_dnn')([cross_out, dnn_out])
    model_desc.add_net('dcn', x.shape, stack_out.shape)
    return stack_out


def autoint_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks.
    """
    if embeddings is None or len(embeddings) <= 0:
        model_desc.add_net('autoint', (None), (None))
        return None
    autoint_emb_concat = Concatenate(axis=1, name='concat_autoint_embedding')(embeddings)
    output = autoint_emb_concat
    for i in range(config.autoint_params['num_attention']):
        output = layers.MultiheadAttention(params=config.autoint_params)(output)
    output = Flatten()(output)
    model_desc.add_net('autoint', autoint_emb_concat.shape, output.shape)
    return output


def fg_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    Feature Generation leverages the strength of CNN to generate local patterns and recombine
     them to generate new features.

    References
    ----------
        .. [1] `Liu B, Tang R, Chen Y, et al. Feature generation by convolutional neural network
        for click-through rate prediction[C]//The World Wide Web Conference. 2019: 1119-1129.`
    """
    if embeddings is None or len(embeddings) <= 0:
        model_desc.add_net('fgcnn', (None), (None))
        return None
    fgcnn_emb_concat = Concatenate(axis=1, name='concat_fgcnn_embedding')(embeddings)
    fg_inputs = tf.expand_dims(fgcnn_emb_concat, axis=-1)
    fg_filters = config.fgcnn_params.get('fg_filters', (14, 16))
    fg_heights = config.fgcnn_params.get('fg_heights', (7, 7))
    fg_pool_heights = config.fgcnn_params.get('fg_pool_heights', (2, 2))
    fg_new_feat_filters = config.fgcnn_params.get('fg_new_feat_filters', (2, 2))
    new_features = list()
    for filters, width, pool, new_filters in zip(fg_filters, fg_heights, fg_pool_heights, fg_new_feat_filters):
        fg_inputs, new_feats = layers.FGCNN(
            filters=filters,
            kernel_height=width,
            pool_height=pool,
            new_filters=new_filters
        )(fg_inputs)
        new_features.append(new_feats)
    concat_all_features = Concatenate(axis=1)(new_features + [fgcnn_emb_concat])
    model_desc.add_net('fg', fgcnn_emb_concat.shape, concat_all_features.shape)
    return concat_all_features


def fgcnn_cin_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    FGCNN with CIN as deep classifier
    """
    fg_output = fg_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc)
    cin_output = layers.CIN(params=config.cin_params)(fg_output)
    model_desc.add_net('fgcnn-cin', fg_output.shape, cin_output.shape)
    return cin_output


def fgcnn_fm_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    FGCNN with FM as deep classifier
    """
    fg_output = fg_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc)
    fm_output = layers.FM(name='fm_fgcnn_layer')(fg_output)
    model_desc.add_net('fgcnn-fm', fg_output.shape, fm_output.shape)
    return fm_output


def fgcnn_afm_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    FGCNN with AFM as deep classifier
    """
    fg_output = fg_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc)
    split_features = tf.split(fg_output, fg_output.shape[1], axis=1)
    afm_output = layers.AFM(params=config.afm_params)(split_features)
    model_desc.add_net('fgcnn-afm', fg_output.shape, afm_output.shape)
    return afm_output


def fgcnn_ipnn_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    FGCNN with IPNN as deep classifier
    """
    fg_output = fg_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc)
    split_features = tf.split(fg_output, fg_output.shape[1], axis=1)
    inner_product = layers.InnerProduct()(split_features)
    # dnn_input = Flatten()(concat_all_features)
    dnn_input = Concatenate()([Flatten()(fg_output), inner_product, dense_layer])
    dnn_out = dnn(dnn_input, config.dnn_params, cellname='fgcnn_ipnn')
    model_desc.add_net('fgcnn-ipnn', fg_output.shape, dnn_out.shape)
    return dnn_out


def fgcnn_dnn_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    FGCNN with DNN as deep classifier
    """
    fg_output = fg_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc)
    dnn_input = Concatenate()([Flatten()(fg_output), dense_layer])
    dnn_out = dnn(dnn_input, config.dnn_params, cellname='fgcnn_dnn')
    model_desc.add_net('fgcnn-ipnn', fg_output.shape, dnn_out.shape)
    return dnn_out


def fibi_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    The SENET layer can convert an embedding layer into the SENET-Like embedding features, which helps
    to boost feature discriminability. The following Bilinear-Interaction layer models second order
    feature interactions on the original embedding and the SENET-Like embedding respectively. Subsequently,
    these cross features are concatenated by a combination layer which merges the outputs of
    Bilinear-Interaction layer.
    """
    if embeddings is None or len(embeddings) <= 0:
        model_desc.add_net('fibi', (None), (None))
        return None
    senet_emb_concat = Concatenate(axis=1, name='concat_senet_embedding')(embeddings)

    senet_pooling_op = config.fibinet_params.get('senet_pooling_op', 'mean')
    senet_reduction_ratio = config.fibinet_params.get('senet_reduction_ratio', 3)
    bilinear_type = config.fibinet_params.get('bilinear_type', 'field_interaction')

    senet_embedding = layers.SENET(pooling_op=senet_pooling_op,
                                   reduction_ratio=senet_reduction_ratio,
                                   name='senet_layer')(senet_emb_concat)
    senet_bilinear_out = layers.BilinearInteraction(bilinear_type=bilinear_type,
                                                    name='senet_bilinear_layer')(senet_embedding)
    bilinear_out = layers.BilinearInteraction(bilinear_type=bilinear_type,
                                              name='embedding_bilinear_layer')(senet_emb_concat)
    concat_bilinear = Concatenate(axis=1, name='concat_bilinear')([senet_bilinear_out, bilinear_out])
    model_desc.add_net('fibi', senet_emb_concat.shape, concat_bilinear.shape)
    return concat_bilinear


def fibi_dnn_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc):
    """
    FiBiNet with DNN as deep classifier
    """
    fibi_output = fibi_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config, model_desc)
    dnn_input = Concatenate(name='concat_bilinear_dense')(
        [Flatten(name='flatten_fibi_output')(fibi_output), dense_layer])
    dnn_out = dnn(dnn_input, config.dnn_params, cellname='fibi_dnn')
    model_desc.add_net('fibi-dnn', fibi_output.shape, dnn_out.shape)
    return dnn_out


def serialize(nets_fn):
    return serialize_keras_object(nets_fn)


def deserialize(name, custom_objects=None):
    return deserialize_keras_object(
        name,
        module_objects=globals(),
        custom_objects=custom_objects,
        printable_module_name='nets function')


def dnn(x, params, cellname='dnn'):
    custom_dnn_fn = params.get('custom_dnn_fn')
    if custom_dnn_fn is not None:
        return custom_dnn_fn(x, params, cellname + '_custom')

    hidden_units = params.get('hidden_units', ((128, 0, True), (64, 0, False)))
    activation = params.get('activation', 'relu')
    kernel_initializer = params.get('kernel_initializer', 'he_uniform')
    kernel_regularizer = params.get('kernel_regularizer')
    activity_regularizer = params.get('activity_regularizer')
    if len(hidden_units) <= 0:
        raise ValueError(
            '[hidden_units] must be a list of tuple([units],[dropout_rate],[use_bn]) and at least one tuple.')
    index = 1
    for units, dropout, batch_norm in hidden_units:
        x = Dense(units, use_bias=not batch_norm, name=f'{cellname}_dense_{index}',
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer,
                  activity_regularizer=activity_regularizer,
                  )(x)
        if batch_norm:
            x = BatchNormalization(name=f'{cellname}_bn_{index}')(x)
        x = Activation(activation=activation, name=f'{cellname}_activation_{index}')(x)
        if dropout > 0:
            x = Dropout(dropout, name=f'{cellname}_dropout_{index}')(x)
        index += 1
    return x


def custom_dnn_D_A_D_B(x, params, cellname='dnn_D_A_D_B'):
    hidden_units = params.get('hidden_units', ((128, 0, True), (64, 0, False)))
    activation = params.get('activation', 'relu')
    kernel_initializer = params.get('kernel_initializer', 'he_uniform')
    kernel_regularizer = params.get('kernel_regularizer')
    activity_regularizer = params.get('activity_regularizer')
    if len(hidden_units) <= 0:
        raise ValueError(
            '[hidden_units] must be a list of tuple([units],[dropout_rate],[use_bn]) and at least one tuple.')
    index = 1
    for units, dropout, batch_norm in hidden_units:
        x = Dense(units,
                  activation=activation,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer,
                  activity_regularizer=activity_regularizer,
                  name=f'{cellname}_dense_{index}')(x)
        if dropout > 0:
            x = Dropout(dropout, name=f'{cellname}_dropout_{index}')(x)
        if batch_norm:
            x = BatchNormalization(name=f'{cellname}_bn_{index}')(x)
        index += 1
    return x


def get(identifier):
    """Returns function.
    Arguments:
        identifier: Function or string
    Returns:
        Nets function denoted by input:
        - Function corresponding to the input string or input function.
    For example:
    >>> nets.get('dnn_nets')
     <function dnnlogit at 0x1222a3d90>
    """
    if identifier is None:
        raise ValueError(f'identifier can not be none.')
    if isinstance(identifier, six.string_types):
        identifier = str(identifier)
        nets_fn = custom_nets.get(identifier)
        if nets_fn is not None:
            return nets_fn
        return deserialize(identifier)
    elif callable(identifier):
        register_nets(identifier)
        return identifier
    else:
        raise TypeError(f'Could not interpret nets function identifier: {repr(identifier)}')


custom_nets = {}


def get_nets(nets):
    str_nets = []
    nets = set(nets)
    for net in nets:
        if isinstance(net, str):
            str_nets.append(net)
        else:
            name = register_nets(net)
            str_nets.append(name)
    return str_nets


def register_nets(nets_fn):
    if not callable(nets_fn):
        raise ValueError('nets_fn must be a valid callable function.')
    if signature(nets_fn) != signature(linear):
        raise ValueError(f'Signature of nets_fn is invalid, except {signature(linear)}  but {signature(nets_fn)}')
    custom_nets[nets_fn.__name__] = nets_fn
    return nets_fn.__name__
