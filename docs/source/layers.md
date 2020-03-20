# Layers

## FM

Factorization Machine to model order-2 feature interactions.

**Call arguments**:
- x: A 3D tensor.

**Input shape**: 
- 3D tensor with shape: (batch_size, field_size, embedding_size)

**Output shape**:
- 2D tensor with shape: (batch_size, 1)

**References**:
- [1] Rendle S. Factorization machines[C]//2010 IEEE International Conference on Data Mining. IEEE, 2010: 995-1000.
- [2] Guo H, Tang R, Ye Y, et al. Deepfm: An end-to-end wide & deep learning framework for CTR prediction[J]. arXiv preprint arXiv:1804.04950, 2018.


## AFM
Attentional Factorization Machine (AFM), which learns the importance of each feature interaction from data via a neural attention network.

**Arguments**:
- hidden_factor: int, (default=16)
- activation_function : str, (default='relu')
- kernel_regularizer : str or object, (default=None)
- dropout_rate: float, (default=0)

**Call arguments**:
- x: A list of 3D tensor.

**Input shape**:
- A list of 3D tensor with shape: (batch_size, 1, embedding_size)

**Output shape**:
- 2D tensor with shape: (batch_size, 1)

**References**:
- [1] Xiao J, Ye H, He X, et al. Attentional factorization machines: Learning the weight of feature interactions via attention networks[J]. arXiv preprint arXiv:1708.04617, 2017.
- [2] [https://github.com/hexiangnan/attentional_factorization_machine](https://github.com/hexiangnan/attentional_factorization_machine)


## CIN
Compressed Interaction Network (CIN), with the following considerations: (1) interactions are applied at vector-wise level, not at bit-wise level; (2) high-order feature interactions is measured explicitly; (3) the complexity of network will not grow exponentially with the degree of interactions.

**Arguments**:
- cross_layer_size: tuple of int, (default = (128, 128,))
- activation: str, (default='relu')
- use_residual: bool, (default=False)
- use_bias: bool, (default=False)
- direct: bool, (default=False)
- reduce_D:bool, (default=False)

**Call arguments**:
- x: A 3D tensor.

**Input shape**:
- A 3D tensor with shape: (batch_size, num_fields, embedding_size)

**Output shape**:
- 2D tensor with shape: (batch_size, *)

**References**:
- [1] Lian J, Zhou X, Zhang F, et al. xdeepfm: Combining explicit and implicit feature interactions for recommender systems[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018: 1754-1763.
- [2] [https://github.com/Leavingseason/xDeepFM](https://github.com/Leavingseason/xDeepFM)



## MultiheadAttention

A multihead self-attentive nets with residual connections to explicitly model the feature interactions.

**Arguments**:
- num_head: int, (default=1)
- dropout_rate: float, (default=0)
- use_residual: bool, (default=True)

**Call arguments**:
- x: A 3D tensor.

**Input shape**:
- 3D tensor with shape: (batch_size, field_size, embedding_size)

**Output shape**:
- 3D tensor with shape: (batch_size, field_size, embedding_size*num_head)

**References**:

- [1] Song W, Shi C, Xiao Z, et al. Autoint: Automatic feature interaction learning via self-attentive neural networks[C]//Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019: 1161-1170.
- [2] [https://github.com/shichence/AutoInt](https://github.com/shichence/AutoInt)

## FGCNN
Feature Generation nets leverages the strength of CNN to generate local patterns and recombine them to generate new features.

**Arguments**:
- filters: int, the filters of convolutional layer
- kernel_height: int, the height of kernel_size of convolutional layer
- new_filters: int, the number of new features' map in recombination layer
- pool_height: int, the height of pool_size of pooling layer
- activation: str, (default='tanh')

**Call arguments**:
- x: A 4D tensor.

**Input shape**:
- 4D tensor with shape: (batch_size, field_size, embedding_size, 1)

**Output shape**:
- pooling_output - 4D tensor
- new_features - 3D tensor with shape: (batch_size, field_size*new_filters, embedding_size)

**References**:
- [1] Liu B, Tang R, Chen Y, et al. Feature generation by convolutional neural network for click-through rate prediction[C]//The World Wide Web Conference. 2019: 1119-1129.

## SENET

SENET layer can dynamically increase the weights of important features and decrease the weights of uninformative features to let the model pay more attention to more important features.

**Arguments**:
- pooling_op: str, (default='mean'). Pooling methods to squeeze the original embedding E into a statistic vector Z
- reduction_ratio: float, (default=3). Hyper-parameter for dimensionality-reduction

**Call arguments**:
- x: A 3D tensor.

**Input shape**:
- 3D tensor with shape: (batch_size, field_size, embedding_size)

**Output shape**:
- 3D tensor with shape: (batch_size, field_size, embedding_size)

**References**:
- [1] Huang T, Zhang Z, Zhang J. FiBiNET: combining feature importance and bilinear feature interaction for click-through rate prediction[C]//Proceedings of the 13th ACM Conference on Recommender Systems. 2019: 169-177.

## BilinearInteraction
The Bilinear-Interaction layer combines the inner product and Hadamard product to learn the feature interactions.

**Arguments**:
- bilinear_type: str, (default='field_interaction'). The type of bilinear functions
    - `field_interaction`
    - `field_all`
    - `field_each`

**Call arguments**:
- x: A 3D tensor.

**Input shape**:
- 3D tensor with shape: (batch_size, field_size, embedding_size)

**Output shape**:
- 3D tensor with shape: (batch_size, *, embedding_size)

**References**:
- [1] Huang T, Zhang Z, Zhang J. FiBiNET: combining feature importance and bilinear feature interaction for click-through rate prediction[C]//Proceedings of the 13th ACM Conference on Recommender Systems. 2019: 169-177.


## Cross
The cross network is composed of cross layers to apply explicit feature crossing in an efficient way.

**Arguments**:
- num_cross_layer: int, (default=2). The number of cross layers

**Call arguments**:
- x: A 2D tensor.

**Input shape**:
- 2D tensor with shape: (batch_size, field_size)

**Output shape**:
- 2D tensor with shape: (batch_size, field_size)

**References**:
- [1] Wang R, Fu B, Fu G, et al. Deep & cross network for ad click predictions[M]//Proceedings of the ADKDD'17. 2017: 1-7.

## InnerProduct
InnerProduct layer used in PNN

**Call arguments**:
- x: A list of 3D tensor.

**Input shape**:
- A list of 3D tensor with shape (batch_size, 1, embedding_size)

**Output shape**:
- 2D tensor with shape: (batch_size, num_fields*(num_fields-1)/2)

**References**:
- [1] Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//2016 IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016: 1149-1154.
- [2] Qu Y, Fang B, Zhang W, et al. Product-based neural networks for user response prediction over multi-field categorical data[J]. ACM Transactions on Information Systems (TOIS), 2018, 37(1): 1-35.
- [3] [https://github.com/Atomu2014/product-nets](https://github.com/Atomu2014/product-nets)

## OuterProduct
OuterProduct layer used in PNN

**Arguments**:
- outer_product_kernel_type: str, (default='mat'). The type of outer product kernel
    - `mat`
    - `vec`
    - `num`
**Call arguments**:
- x: A list of 3D tensor.

**Input shape**:
- A list of 3D tensor with shape (batch_size, 1, embedding_size)

**Output shape**:
- 2D tensor with shape: (batch_size, num_fields*(num_fields-1)/2)

**References**:
- [1] Qu Y, Cai H, Ren K, et al. Product-based neural networks for user response prediction[C]//2016 IEEE 16th International Conference on Data Mining (ICDM). IEEE, 2016: 1149-1154.
- [2] Qu Y, Fang B, Zhang W, et al. Product-based neural networks for user response prediction over multi-field categorical data[J]. ACM Transactions on Information Systems (TOIS), 2018, 37(1): 1-35.
- [3] [https://github.com/Atomu2014/product-nets](https://github.com/Atomu2014/product-nets)



