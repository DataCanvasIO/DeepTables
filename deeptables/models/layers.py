# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization, Activation, Concatenate, Flatten, Input, \
    Embedding, Lambda, Add, Conv2D, MaxPooling2D, SpatialDropout1D
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, losses, constraints
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.keras.metrics import RootMeanSquaredError
import itertools

from ..utils import dt_logging, consts, gpu

logger = dt_logging.get_logger()

class FM(Layer):
    """Factorization Machine to model order-2 feature interactions
    Arguments:

    Call arguments:
        x: A 3D tensor.

    Input shape
    -----------
        - 3D tensor with shape:
         `(batch_size, field_size, embedding_size)`

    Output shape
    ------------
        - 2D tensor with shape:
         `(batch_size, 1)`

    References
    ----------
    .. [1] `Rendle S. Factorization machines[C]//2010 IEEE International Conference on Data Mining. IEEE, 2010: 995-1000.`
    .. [2] `Guo H, Tang R, Ye Y, et al. Deepfm: An end-to-end wide & deep learning framework for CTR prediction[J]. arXiv preprint arXiv:1804.04950, 2018.`
    """

    def __init__(self, **kwargs):
        super(FM, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        if K.ndim(x) != 3:
            raise ValueError(f'Wrong dimensions of inputs, expected 3 but input {K.ndim(x)}.')
        square_of_sum = tf.square(tf.reduce_sum(
            x, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(
            x * x, axis=1, keepdims=True)
        cross = square_of_sum - sum_of_square
        cross = 0.5 * tf.reduce_sum(cross, axis=2, keepdims=False)
        return cross


class MultiheadAttention(Layer):
    """ A multihead self-attentive nets with residual connections to explicitly model the
    feature interactions.

    Arguments:
        params: dict
        ------
            - num_head: int, (default=1)
            - dropout_rate: float, (default=0)
            - use_residual: bool, (default=True)

    Call arguments:
        x: A 3D tensor.

    Input shape
    -----------
        - 3D tensor with shape:
         `(batch_size, field_size, embedding_size)`

    Output shape
    ------------
        - 3D tensor with shape:
         `(batch_size, field_size, embedding_size*num_head)`

    References
    ----------
    .. [1] `Song W, Shi C, Xiao Z, et al. Autoint: Automatic feature interaction learning via
    self-attentive neural networks[C]//Proceedings of the 28th ACM International Conference on
    Information and Knowledge Management. 2019: 1161-1170.`
    .. [2] https://github.com/shichence/AutoInt
    """

    def __init__(self, params, **kwargs):
        self.params = params
        self.num_heads = params.get('num_heads', 1)
        self.dropout_rate = params.get('dropout_rate', 0)
        self.use_residual = params.get('use_residual', True)
        super(MultiheadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_units = input_shape[-1]
        self.dense_Q = Dense(self.num_units, activation='relu', kernel_initializer='he_uniform')
        self.dense_K = Dense(self.num_units, activation='relu', kernel_initializer='he_uniform')
        self.dense_V = Dense(self.num_units, activation='relu', kernel_initializer='he_uniform')

        self.dense_residual = Dense(self.num_units, activation='relu', kernel_initializer='he_uniform')
        self.dropout_weights = Dropout(rate=self.dropout_rate)
        self.batch_normalize = BatchNormalization()
        super(MultiheadAttention, self).build(input_shape)

    def call(self, x, **kwargs):
        if K.ndim(x) != 3:
            raise ValueError(f'Wrong dimensions of inputs, expected 3 but input {K.ndim(x)}.')

        # Linear projections
        queries = x
        keys = x
        values = x
        q = self.dense_Q(queries)
        k = self.dense_K(keys)
        v = self.dense_V(values)
        if self.use_residual:
            V_res = self.dense_residual(values)

        # Split and concat
        Q_ = tf.concat(tf.split(q, self.num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(k, self.num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(v, self.num_heads, axis=2), axis=0)

        # Multiplication
        weights = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))
        # Scale
        weights = weights / (K_.get_shape().as_list()[-1] ** 0.5)
        # Activation
        weights = tf.nn.softmax(weights)
        # Dropouts
        weights = self.dropout_weights(weights)
        # Weighted sum
        outputs = tf.matmul(weights, V_)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)

        # Residual connection
        if self.use_residual:
            outputs += V_res
        outputs = tf.nn.relu(outputs)
        # Normalize
        outputs = self.batch_normalize(outputs)
        return outputs

    def get_config(self, ):
        config = {'params': self.params}
        base_config = super(MultiheadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class FGCNN(Layer):
    """Feature Generation nets leverages the strength of CNN to generate local patterns
    and recombine them to generate new features.

        Arguments:
            filters: int
                the filters of convolutional layer
            kernel_height
                the height of kernel_size of convolutional layer
            new_filters
                the number of new features' map in recombination layer
            pool_height
                the height of pool_size of pooling layer
            activation: str, (default='tanh')

        Call arguments:
            x: A 4D tensor.

        Input shape
        -----------
            - 4D tensor with shape:
             `(batch_size, field_size, embedding_size, 1)`

        Output shape
        ------------
            pooling_output - 4D tensor
            new_features - 3D tensor with shape:
             `(batch_size, field_size*new_filters, embedding_size)`

        References
        ----------
        .. [1] `Liu B, Tang R, Chen Y, et al. Feature generation by convolutional neural network
        for click-through rate prediction[C]//The World Wide Web Conference. 2019: 1119-1129.`
    """

    def __init__(self, filters, kernel_height, new_filters, pool_height, activation='tanh', **kwargs):
        self.filters = filters
        self.kernel_height = kernel_height
        self.new_filters = new_filters
        self.pool_height = pool_height
        self.activation = activation
        super(FGCNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.conv2d = Conv2D(
            filters=self.filters,
            strides=(1, 1),
            kernel_size=(self.kernel_height, 1),
            padding='same',
            activation=self.activation,
            kernel_initializer='glorot_uniform',
            use_bias=True
        )
        self.maxpooling2d = MaxPooling2D(pool_size=(self.pool_height, 1))
        self.flatten = Flatten()
        self.dense_output = Dense(
            units=input_shape[1] * input_shape[2] * self.new_filters,
            activation=self.activation,
            kernel_initializer='glorot_uniform',
            use_bias=True
        )

    def call(self, x, **kwargs):
        output = x
        output = self.conv2d(output)
        embedding_size = output.shape[2]
        pooling_output = self.maxpooling2d(output)
        new_features = self.flatten(pooling_output)
        new_features = self.dense_output(new_features)
        new_features = tf.reshape(new_features,
                                  shape=(-1, output.shape[1] * self.new_filters, embedding_size))
        return pooling_output, new_features

    def get_config(self, ):
        config = {'filters': self.filters,
                  'kernel_width': self.kernel_height,
                  'new_filters': self.new_filters,
                  'pool_width': self.pool_height,
                  'activation': self.activation, }
        base_config = super(FGCNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SENET(Layer):
    """SENET layer can dynamically increase the weights of important features and decrease the weights
    of uninformative features to let the model pay more attention to more important features.

        Arguments:
            pooling_op: str, (default='mean')
                pooling methods to squeeze the original embedding E into a statistic vector Z
                - mean
                - max
            reduction_ratio: float, (default=3)
                hyper-parameter for dimensionality-reduction

        Call arguments:
            x: A 3D tensor.

        Input shape
        -----------
            - 3D tensor with shape:
             `(batch_size, field_size, embedding_size)`

        Output shape
        ------------
            - 3D tensor with shape:
             `(batch_size, field_size, embedding_size)`

        References
        ----------
        .. [1] `Huang T, Zhang Z, Zhang J. FiBiNET: combining feature importance and bilinear feature
        interaction for click-through rate prediction[C]//Proceedings of the 13th ACM Conference on
        Recommender Systems. 2019: 169-177.`
    """

    def __init__(self, pooling_op='mean', reduction_ratio=3, **kwargs):
        self.pooling_op = pooling_op
        self.reduction_ratio = reduction_ratio
        super(SENET, self).__init__(**kwargs)

    def build(self, input_shape):
        self.field_num = input_shape[1]
        self.embedding_size = input_shape[-1]
        self.reduction_num = max(self.field_num // self.reduction_ratio, 1)
        self.dense_att1 = Dense(units=self.reduction_num, activation='relu', kernel_initializer='he_uniform')
        self.dense_att2 = Dense(units=self.field_num, activation='relu', kernel_initializer='he_uniform')
        super(SENET, self).build(input_shape)

    def call(self, x, training=None, **kwargs):
        if K.ndim(x) != 3:
            raise ValueError(f'Wrong dimensions of inputs, expected 3 but input {K.ndim(x)}.')
        # inputs = concat_func(inputs, axis=1)
        if self.pooling_op == 'max':
            Z = tf.reduce_max(x, axis=-1)
        else:
            Z = tf.reduce_mean(x, axis=-1)
        A1 = self.dense_att1(Z)
        A2 = self.dense_att2(A1)
        V = tf.multiply(x, tf.expand_dims(A2, axis=2))
        return V

    def get_config(self, ):
        config = {'reduction_ratio': self.reduction_ratio,
                  'pooling_op': self.pooling_op
                  }
        base_config = super(SENET, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BilinearInteraction(Layer):
    """The Bilinear-Interaction layer combines the inner product and Hadamard product to learn the
    feature interactions.

    Arguments:
        bilinear_type: str, (default='field_interaction')
            the type of bilinear functions
            - field_interaction
            - field_all
            - field_each

    Call arguments:
        x: A 3D tensor.

    Input shape
    -----------
        - 3D tensor with shape:
         `(batch_size, field_size, embedding_size)`

    Output shape
    ------------
        - 3D tensor with shape:
         `(batch_size, *, embedding_size)`

    References
    ----------
    .. [1] `Huang T, Zhang Z, Zhang J. FiBiNET: combining feature importance and bilinear feature
    interaction for click-through rate prediction[C]//Proceedings of the 13th ACM Conference on
    Recommender Systems. 2019: 169-177.`
    """

    def __init__(self, bilinear_type='field_interaction', **kwargs):
        self.bilinear_type = bilinear_type
        super(BilinearInteraction, self).__init__(**kwargs)

    def build(self, input_shape):
        embedding_size = input_shape[-1]
        self.field_num = input_shape[1]
        if self.bilinear_type == "field_all":
            self.W = self.add_weight(shape=(embedding_size, embedding_size), name="bilinear_weight")
        elif self.bilinear_type == "field_each":
            self.W_list = [self.add_weight(shape=(embedding_size, embedding_size), name="bilinear_weight" + str(i))
                           for
                           i in range(self.field_num - 1)]
        else:  # "field_interaction":
            self.W_list = [
                self.add_weight(shape=(embedding_size, embedding_size),
                                name="bilinear_weight" + str(i) + '_' + str(j))
                for i, j in
                itertools.combinations(range(self.field_num), 2)]

        super(BilinearInteraction, self).build(input_shape)

    def call(self, x, **kwargs):
        if K.ndim(x) != 3:
            raise ValueError(f'Wrong dimensions of inputs, expected 3 but input {K.ndim(x)}.')
        x = tf.split(x, self.field_num, axis=1)
        if self.bilinear_type == "field_all":
            p = [tf.tensordot(v_i, self.W, axes=(-1, 0)) * v_j for v_i, v_j in itertools.combinations(x, 2)]
        elif self.bilinear_type == "field_each":
            p = [tf.tensordot(x[i], self.W_list[i], axes=(-1, 0)) * x[j] for i, j in
                 itertools.combinations(range(len(x)), 2)]
        else:  # "field_interaction":
            p = [tf.tensordot(v[0], w, axes=(-1, 0)) * v[1] for v, w in
                 zip(itertools.combinations(x, 2), self.W_list)]
        concat_all = Concatenate(axis=1)(p)
        return concat_all

    def get_config(self, ):
        config = {'bilinear_type': self.bilinear_type}
        base_config = super(BilinearInteraction, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Cross(Layer):
    """The cross network is composed of cross layers to apply explicit feature crossing in an
    efficient way.

    Arguments:
        num_cross_layer: int, (default=2)
            the number of cross layers

    Call arguments:
        x: A 2D tensor.

    Input shape
    -----------
        - 2D tensor with shape:
         `(batch_size, field_size)`

    Output shape
    ------------
        - 2D tensor with shape:
         `(batch_size, field_size)`

    References
    ----------
    .. [1] `Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[M]//Proceedings
    of the ADKDD'17. 2017: 1-7.`
    """

    def __init__(self, params, **kwargs):
        self.params = params
        self.num_cross_layer = params.get('num_cross_layer', 2)
        super(Cross, self).__init__(**kwargs)

    def build(self, input):
        num_dims = input[-1]
        self.kernels = []
        self.bias = []
        for i in range(self.num_cross_layer):
            self.kernels.append(
                self.add_weight(name='kernels_' + str(i), shape=(num_dims, 1), initializer='glorot_uniform',
                                trainable=True))
            self.bias.append(
                self.add_weight(name='bias_' + str(i), shape=(num_dims, 1), initializer='zeros', trainable=True))

    def call(self, x, **kwargs):
        if K.ndim(x) != 2:
            raise ValueError(f'Wrong dimensions of x, expected 2 but input {K.ndim(x)}.')
        x_f = tf.expand_dims(x, axis=-1)
        x_n = x_f
        for i in range(self.num_cross_layer):
            x_n = tf.matmul(x_f, tf.tensordot(x_n, self.kernels[i], axes=(1, 0))) + x_n + self.bias[i]
        x_n = tf.reshape(x_n, (-1, x_f.shape[1]))
        return x_n

    def get_config(self, ):
        config = {'params': self.params}
        base_config = super(Cross, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class InnerProduct(Layer):
    """Inner-Product layer

    Arguments:

    Call arguments:
        x: A list of 3D tensor.

    Input shape
    -----------
        - A list of 3D tensor with shape (batch_size, 1, embedding_size)

    Output shape
    ------------
        - 2D tensor with shape:
         `(batch_size, num_fields*(num_fields-1)/2)`

    References
    ----------
    .. [1] `Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//2016
    IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016: 1149-1154.`
    .. [2] `Qu Y, Fang B, Zhang W, et al. Product-based neural networks for user response prediction over
    multi-field categorical datasets[J]. ACM Transactions on Information Systems (TOIS), 2018, 37(1): 1-35.`
    .. [3] https://github.com/Atomu2014/product-nets
    """

    def __init__(self, **kwargs):
        super(InnerProduct, self).__init__(**kwargs)

    def call(self, x, **kwargs):
        if K.ndim(x[0]) != 3:
            raise ValueError(f'Wrong dimensions of inputs, expected 3 but input {K.ndim(x[0])}.')
        num_inputs = len(x)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        row = []
        col = []
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = tf.concat([x[i] for i in row], axis=1)
        q = tf.concat([x[j] for j in col], axis=1)
        ip = tf.reshape(tf.reduce_sum(p * q, [-1]), [-1, num_pairs])
        return ip

    def get_config(self):
        return super(InnerProduct, self).get_config()


class OuterProduct(Layer):
    """Outer-Product layer

    Arguments:
        outer_product_kernel_type: str, (default='mat')
            the type of outer product kernel
            - mat
            - vec
            - num

    Call arguments:
        x: A list of 3D tensor.

    Input shape
    -----------
        - A list of 3D tensor with shape (batch_size, 1, embedding_size)

    Output shape
    ------------
        - 2D tensor with shape:
         `(batch_size, num_fields*(num_fields-1)/2)`

    References
    ----------
    .. [1] `Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//2016
    IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016: 1149-1154.`
    .. [2] `Qu Y, Fang B, Zhang W, et al. Product-based neural networks for user response prediction over
    multi-field categorical datasets[J]. ACM Transactions on Information Systems (TOIS), 2018, 37(1): 1-35.`
    .. [3] https://github.com/Atomu2014/product-nets
    """

    def __init__(self, params, **kwargs):
        self.params = params
        self.kernel_type = params.get('outer_product_kernel_type', 'mat')
        if self.kernel_type not in ['mat', 'vec', 'num']:
            raise ValueError("kernel_type must be mat,vec or num")
        super(OuterProduct, self).__init__(**kwargs)

    def build(self, input):
        num_inputs = len(input)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        embed_size = input[0][-1]
        if self.kernel_type == 'mat':
            self.kernel = self.add_weight(shape=(embed_size, num_pairs, embed_size), name='kernel')
        elif self.kernel_type == 'vec':
            self.kernel = self.add_weight(shape=(num_pairs, embed_size,), name='kernel')
        elif self.kernel_type == 'num':
            self.kernel = self.add_weight(shape=(num_pairs, 1), name='kernel')
        super(OuterProduct, self).build(input)

    def call(self, x, **kwargs):
        if K.ndim(x[0]) != 3:
            raise ValueError(f'Wrong dimensions of inputs, expected 3 but input {K.ndim(x[0])}.')
        row = []
        col = []
        num_inputs = len(x)
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)
        p = tf.concat([x[i] for i in row], axis=1)
        q = tf.concat([x[i] for i in col], axis=1)

        # -------------------------
        if self.kernel_type == 'mat':
            p = tf.expand_dims(p, 1)
            # k     k* pair* k
            # batch * pair
            kp = tf.reduce_sum(
                # batch * pair * k
                tf.multiply(
                    # batch * pair * k
                    tf.transpose(
                        # batch * k * pair
                        tf.reduce_sum(
                            # batch * k * pair * k
                            tf.multiply(
                                p, self.kernel),
                            -1),
                        [0, 2, 1]),
                    q),
                -1)
        else:
            # 1 * pair * (k or 1)
            k = tf.expand_dims(self.kernel, 0)
            # batch * pair
            kp = tf.reduce_sum(p * q * k, -1)
            # p q # b * p * k
        return kp

    def get_config(self, ):
        config = {'params': self.params}
        base_config = super(OuterProduct, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CIN(Layer):
    """ Compressed Interaction Network (CIN), with the following considerations: (1) interactions
    are applied at vector-wise level, not at bit-wise level; (2) high-order feature interactions
    is measured explicitly; (3) the complexity of network will not grow exponentially with the degree
    of interactions.

    Arguments:
        cross_layer_size: tuple of int, (default = (128, 128,))
        activation: str, (default='relu')
        use_residual: bool, (default=False)
        use_bias: bool, (default=False)
        direct: bool, (default=False)
        reduce_D:bool, (default=False)

    Call arguments:
        x: A 3D tensor.

    Input shape
    -----------
        - A 3D tensor with shape:
         `(batch_size, num_fields, embedding_size)`

    Output shape
    ------------
        - 2D tensor with shape:
         `(batch_size, *)`

    References
    ----------
    .. [1] `Lian J, Zhou X, Zhang F, et al. xdeepfm: Combining explicit and implicit feature interactions
    for recommender systems[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge
    Discovery & Data Mining. 2018: 1754-1763.`
    .. [2] https://github.com/Leavingseason/xDeepFM
    """

    def __init__(self, params, **kwargs):
        self.params = params
        self.cross_layer_size = params.get('cross_layer_size', (128, 128,))
        self.activation = params.get('activation', 'relu')
        self.use_residual = params.get('use_residual', False)
        self.use_bias = params.get('use_bias', False)
        self.direct = params.get('direct', False)
        self.reduce_D = params.get('reduce_D', False)

        if len(self.cross_layer_size) == 0:
            raise ValueError(
                "cross_layer_size must be a list(tuple) of length greater than 1")
        super(CIN, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 3 dimensions" % (len(input_shape)))

        self.field_nums = [int(input_shape[1])]
        embed_dim = input_shape[-1]
        self.f_ = []
        self.f0_ = []
        self.f__ = []
        self.bias = []
        for i, layer_size in enumerate(self.cross_layer_size):
            if self.reduce_D:
                self.f0_.append(
                    self.add_weight(name=f'f0_{i}', shape=[1, layer_size, self.field_nums[0], embed_dim],
                                    dtype=tf.float32, initializer='he_uniform'))
                self.f__.append(
                    self.add_weight(name=f'f__{i}', shape=[1, layer_size, embed_dim, self.field_nums[-1]],
                                    dtype=tf.float32, initializer='he_uniform'))
            else:
                self.f_.append(
                    self.add_weight(name=f'f_{i}', shape=[1, self.field_nums[-1] * self.field_nums[0], layer_size],
                                    dtype=tf.float32, initializer='he_uniform'))
            if self.use_bias:
                self.bias.append(self.add_weight(name=f'bias{i}', shape=[layer_size], dtype=tf.float32,
                                                 initializer='zeros'))

            if self.direct:
                self.field_nums.append(layer_size)
            else:
                if i != len(self.cross_layer_size) - 1 and layer_size % 2 > 0:
                    raise ValueError(
                        "cross_layer_size must be even number except for the last layer when direct=True")
                self.field_nums.append(layer_size // 2)
        self.activation_layers = [Activation(self.activation) for _ in self.cross_layer_size]
        if self.use_residual:
            self.exFM_out0 = Dense(self.cross_layer_size[-1], activation=self.activation,
                                   kernel_initializer='he_uniform')
            self.exFM_out = Dense(1, activation=None)
        else:
            self.exFM_out = Dense(1, activation=None)

        super(CIN, self).build(input_shape)

    def call(self, x, **kwargs):
        if K.ndim(x) != 3:
            raise ValueError(f'Wrong dimensions of inputs, expected 3 but input {K.ndim(x)}.')

        dim = int(x.get_shape()[-1])
        hidden_nn_layers = [x]
        final_result = []

        split_tensor0 = tf.split(hidden_nn_layers[0], dim * [1], 2)
        for idx, layer_size in enumerate(self.cross_layer_size):
            split_tensor = tf.split(hidden_nn_layers[-1], dim * [1], 2)
            dot_result_m = tf.matmul(split_tensor0, split_tensor, transpose_b=True)
            dot_result_o = tf.reshape(dot_result_m, shape=[dim, -1, self.field_nums[0] * self.field_nums[idx]])
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])

            if self.reduce_D:
                f0_ = self.f0_[idx]
                f__ = self.f__[idx]
                f_m = tf.matmul(f0_, f__)
                f_o = tf.reshape(f_m, shape=[1, layer_size, self.field_nums[0] * self.field_nums[idx]])
                filters = tf.transpose(f_o, perm=[0, 2, 1])
            else:
                filters = self.f_[idx]
            curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
            if self.use_bias:
                curr_out = tf.nn.bias_add(curr_out, self.bias[idx])

            curr_out = self.activation_layers[idx](curr_out)
            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])

            if self.direct:
                direct_connect = curr_out
                next_hidden = curr_out
            else:
                if idx != len(self.cross_layer_size) - 1:
                    next_hidden, direct_connect = tf.split(curr_out, 2 * [layer_size // 2], 1)
                else:
                    direct_connect = curr_out
                    next_hidden = 0

            final_result.append(direct_connect)
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)
        result = tf.reduce_sum(result, -1)

        if self.use_residual:
            exFM_out0 = self.exFM_out0(result)
            exFM_in = tf.concat([exFM_out0, result], axis=1)
            exFM_out = self.exFM_out(exFM_in)
        else:
            exFM_out = self.exFM_out(result)
        return exFM_out

    def get_config(self, ):
        config = {'params': self.params}
        base_config = super(CIN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AFM(Layer):
    """Attentional Factorization Machine (AFM), which learns the importance of each feature interaction
    from datasets via a neural attention network.

    Arguments:
        hidden_factor: int, (default=16)
        activation_function : str, (default='relu')
        kernel_regularizer : str or object, (default=None)
        dropout_rate: float, (default=0)

    Call arguments:
        x: A list of 3D tensor.

    Input shape
    -----------
        - A list of 3D tensor with shape: (batch_size, 1, embedding_size)

    Output shape
    ------------
        - 2D tensor with shape:
         `(batch_size, 1)`

    References
    ----------
    .. [1] `Xiao J, Ye H, He X, et al. Attentional factorization machines: Learning the weight of feature
    interactions via attention networks[J]. arXiv preprint arXiv:1708.04617, 2017.`
    .. [2] https://github.com/hexiangnan/attentional_factorization_machine
    """

    def __init__(self, params, **kwargs):
        self.params = params
        self.hidden_factor = params.get('hidden_factor', 16)
        self.dropout_rate = params.get('dropout_rate', 0)
        self.activation_function = params.get('activation', 'relu')
        self.kernel_regularizer = params.get('kernel_regularizer', None)
        super(AFM, self).__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) < 2:
            raise ValueError('A `AttentionalFM` layer should be called '
                             'on a list of at least 2 inputs')
        self.dense_attention = Dense(self.hidden_factor, activation=self.activation_function,
                                     kernel_regularizer=self.kernel_regularizer,
                                     kernel_initializer='glorot_normal', name='dense_afm_attention')
        self.dense_out = Dense(1, use_bias=False)
        self.attention_p = self.add_weight(shape=(self.hidden_factor, 1), name="projection_h")
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def call(self, x, **kwargs):
        if K.ndim(x[0]) != 3:
            raise ValueError(f'Wrong dimensions of inputs, expected 3 but input {K.ndim(x[0])}.')
        row = []
        col = []
        for r, c in itertools.combinations(x, 2):
            row.append(r)
            col.append(c)
        p = tf.concat(row, axis=1)
        q = tf.concat(col, axis=1)
        bi_interaction = p * q

        attention_2 = self.dense_attention(bi_interaction)
        attention_score = tf.nn.softmax(tf.tensordot(attention_2, self.attention_p, axes=(-1, 0)), 1)
        attention_out = tf.reduce_sum(attention_score * bi_interaction, axis=1)
        attention_out = self.dropout(attention_out)
        afm_out = self.dense_out(attention_out)
        return afm_out

    def get_config(self, ):
        config = {'params': self.params}
        base_config = super(AFM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MultiColumnEmbedding(Layer):
    '''
    This class is adapted from tensorflow's implementation of Embedding
    We modify the code to make it suitable for multiple variables from different column in one input.
    '''

    def __init__(self,
                 input_dims,
                 output_dims,
                 dropout_rate=0.,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 **kwargs):
        if 'input_shape' not in kwargs:
            kwargs['input_shape'] = (1,)
        dtype = kwargs.pop('dtype', K.floatx())
        kwargs['autocast'] = False
        super(MultiColumnEmbedding, self).__init__(dtype=dtype, **kwargs)

        if not isinstance(input_dims, (list, tuple)):
            raise ValueError(f'[input_dims] must be a list or tuple.')
        if not isinstance(output_dims, (list, tuple)):
            raise ValueError(f'[output_dims] must be a list or tuple.')
        if len(input_dims) != len(output_dims):
            raise ValueError(f'The length of [input_dims] and [output_dims] must be the same.')
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.dropout_rate = dropout_rate
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero

    def build(self, input_shape):
        if input_shape[1] == 0:
            return
        if input_shape[1] != len(self.input_dims):
            raise ValueError('The inputs dimension on axis 1 must be the same as the length of [input_dims].')

        if context.executing_eagerly() and context.context().num_gpus():
            with ops.device('cpu:0'):
                self.embeddings = []
                for i, (input_dim, output_dim) in enumerate(zip(self.input_dims, self.output_dims)):
                    self.embeddings.append(self.add_weight(
                        shape=(input_dim, output_dim),
                        initializer=self.embeddings_initializer,
                        name=f'embeddings_{i}',
                        regularizer=self.embeddings_regularizer,
                        constraint=self.embeddings_constraint))
        else:
            self.embeddings = []
            for i, (input_dim, output_dim) in enumerate(zip(self.input_dims, self.output_dims)):
                self.embeddings.append(self.add_weight(
                    shape=(input_dim, output_dim),
                    initializer=self.embeddings_initializer,
                    name=f'embeddings_{i}',
                    regularizer=self.embeddings_regularizer,
                    constraint=self.embeddings_constraint))
        self.dropouts = []
        if self.dropout_rate > 0:
            self.dropouts = [SpatialDropout1D(self.dropout_rate) for i in range(len(self.input_dims))]
        self.built = True

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None

        return math_ops.not_equal(inputs, 0)

    def call(self, inputs):
        if inputs.shape[1] == 0:
            return []
        dtype = K.dtype(inputs)
        if dtype != 'int32' and dtype != 'int64':
            inputs = math_ops.cast(inputs, 'int32')
        columns = tf.split(inputs, len(self.embeddings), axis=1)
        out = []
        for i, col in enumerate(columns):
            emb = embedding_ops.embedding_lookup(self.embeddings[i], col)
            if self.dropout_rate > 0:
                emb = self.dropouts[i](emb)
            out.append(emb)
        return out

    def get_config(self):
        config = {
            'input_dims': self.input_dims,
            'output_dims': self.output_dims,
            'dropout_rate': self.dropout_rate,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer':
                regularizers.serialize(self.embeddings_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'embeddings_constraint':
                constraints.serialize(self.embeddings_constraint),
            'mask_zero': self.mask_zero,
        }
        base_config = super(MultiColumnEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BinaryFocalLoss(losses.Loss):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        https://arxiv.org/pdf/1708.02002.pdf
        https://github.com/umbertogriffo/focal-loss-keras
    Usage:
        model.compile(loss=[BinaryFocalLoss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def __init__(self, gamma=2., alpha=.25, reduction=losses.Reduction.AUTO, name='focal_loss'):
        super(BinaryFocalLoss, self).__init__(reduction=reduction, name=name)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(self.alpha * K.pow(1. - pt_1, self.gamma) * K.log(pt_1)) \
               - K.mean((1 - self.alpha) * K.pow(pt_0, self.gamma) * K.log(1. - pt_0))

    def get_config(self):
        config = {'gamma': self.gamma, 'alpha': self.alpha}
        base_config = super(BinaryFocalLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CategoricalFocalLoss(losses.Loss):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://github.com/umbertogriffo/focal-loss-keras
    Usage:
     model.compile(loss=[CategoricalFocalLoss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """

    def __init__(self, gamma=2., alpha=.25, reduction=losses.Reduction.AUTO, name='focal_loss'):
        """Focal loss for multi-classification
        FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
        Notice: y_pred is probability after softmax
        gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
        d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
        Focal Loss for Dense Object Detection
        https://arxiv.org/abs/1708.02002

        Keyword Arguments:
            gamma {float} -- (default: {2.0})
            alpha {float} -- (default: {4.0})
        """
        super(BinaryFocalLoss, self).__init__(reduction=reduction, name=name)
        self.gamma = float(gamma)
        self.alpha = float(alpha)

    def call(self, y_true, y_pred):
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * K.log(y_pred)

        # Calculate Focal Loss
        loss = self.alpha * K.pow(1. - y_pred, self.gamma) * cross_entropy

        # Sum the losses in mini_batch
        return K.sum(loss, axis=1)

    def get_config(self):
        config = {'gamma': self.gamma, 'alpha': self.alpha}
        base_config = super(BinaryFocalLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GHMCLoss:
    def __init__(self, bins=10, momentum=0.75):
        self.bins = bins
        self.momentum = momentum
        self.edges_left, self.edges_right = self.get_edges(
            self.bins)  # edges_left: [bins, 1, 1], edges_right: [bins, 1, 1]
        if momentum > 0:
            self.acc_sum = self.get_acc_sum(self.bins)  # [bins]

    def get_edges(self, bins):
        edges_left = [float(x) / bins for x in range(bins)]
        edges_left = tf.constant(edges_left)  # [bins]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1]
        edges_left = tf.expand_dims(edges_left, -1)  # [bins, 1, 1]

        edges_right = [float(x) / bins for x in range(1, bins + 1)]
        edges_right[-1] += 1e-6
        edges_right = tf.constant(edges_right)  # [bins]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1]
        edges_right = tf.expand_dims(edges_right, -1)  # [bins, 1, 1]
        return edges_left, edges_right

    def get_acc_sum(self, bins):
        acc_sum = [0.0 for _ in range(bins)]
        return tf.Variable(acc_sum, trainable=False)

    def calc(self, input, target, mask=None, is_mask=False):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        mask [batch_num, class_num]
        """
        edges_left, edges_right = self.edges_left, self.edges_right
        mmt = self.momentum
        # gradient length
        self.g = tf.abs(tf.sigmoid(input) - target)  # [batch_num, class_num]
        g = tf.expand_dims(self.g, axis=0)  # [1, batch_num, class_num]
        g_greater_equal_edges_left = tf.greater_equal(g, edges_left)  # [bins, batch_num, class_num]
        g_less_edges_right = tf.less(g, edges_right)  # [bins, batch_num, class_num]
        zero_matrix = tf.cast(tf.zeros_like(g_greater_equal_edges_left),
                              dtype=tf.float32)  # [bins, batch_num, class_num]
        if is_mask:
            mask_greater_zero = tf.greater(mask, 0)
            inds = tf.cast(tf.logical_and(tf.logical_and(g_greater_equal_edges_left, g_less_edges_right),
                                          mask_greater_zero), dtype=tf.float32)  # [bins, batch_num, class_num]
            tot = tf.maximum(tf.reduce_sum(tf.cast(mask_greater_zero, dtype=tf.float32)), 1.0)
        else:
            inds = tf.cast(tf.logical_and(g_greater_equal_edges_left, g_less_edges_right),
                           dtype=tf.float32)  # [bins, batch_num, class_num]
            input_shape = tf.shape(input)
            tot = tf.maximum(tf.cast(input_shape[0] * input_shape[1], dtype=tf.float32), 1.0)
        num_in_bin = tf.reduce_sum(inds, axis=[1, 2])  # [bins]
        num_in_bin_greater_zero = tf.greater(num_in_bin, 0)  # [bins]
        num_valid_bin = tf.reduce_sum(tf.cast(num_in_bin_greater_zero, dtype=tf.float32))

        # num_in_bin = num_in_bin + 1e-12
        if mmt > 0:
            update = tf.compat.v1.assign(self.acc_sum, tf.where(num_in_bin_greater_zero, mmt * self.acc_sum \
                                                                + (1 - mmt) * num_in_bin, self.acc_sum))
            with tf.control_dependencies([update]):
                self.acc_sum_tmp = tf.identity(self.acc_sum, name='updated_accsum')
                acc_sum = tf.expand_dims(self.acc_sum_tmp, -1)  # [bins, 1]
                acc_sum = tf.expand_dims(acc_sum, -1)  # [bins, 1, 1]
                acc_sum = acc_sum + zero_matrix  # [bins, batch_num, class_num]
                weights = tf.where(tf.equal(inds, 1), tot / acc_sum, zero_matrix)
                weights = tf.reduce_sum(weights, axis=0)
        else:
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1]
            num_in_bin = tf.expand_dims(num_in_bin, -1)  # [bins, 1, 1]
            num_in_bin = num_in_bin + zero_matrix  # [bins, batch_num, class_num]
            weights = tf.where(tf.equal(inds, 1), tot / num_in_bin, zero_matrix)
            weights = tf.reduce_sum(weights, axis=0)
        weights = weights / num_valid_bin
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=input)
        loss = tf.reduce_sum(loss * weights) / tot
        return loss

def register_custom_objects(objs_dict:dict):
    for k,v in objs_dict.items():
        if dt_custom_objects.get(k) is None:
            dt_custom_objects[k] = v
        else:
            logger.error(f'`register_custom_objects` cannot register an existing key [{k}].')

dt_custom_objects = {
    'FM': FM,
    'Cross': Cross,
    'InnerProduct': InnerProduct,
    'OuterProduct': OuterProduct,
    'AFM': AFM,
    'MultiheadAttention': MultiheadAttention,
    'FGCNN': FGCNN,
    'BilinearInteraction': BilinearInteraction,
    'SENET': SENET,
    'CIN': CIN,
    'MultiColumnEmbedding': MultiColumnEmbedding,
    'CategoricalFocalLoss': CategoricalFocalLoss,
    'BinaryFocalLoss': BinaryFocalLoss,
}
