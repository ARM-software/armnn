# TensorFlow operators that the Arm NN SDK supports

This reference guide provides a list of TensorFlow operators the Arm NN SDK currently supports.

The Arm NN SDK TensorFlow parser currently only supports fp32 operators.

## Fully supported

**avg_pool**

See the TensorFlow [avg_pool documentation](https://www.tensorflow.org/api_docs/python/tf/nn/avg_pool) for more information.

**bias_add**

See the TensorFlow [bias_add documentation](https://www.tensorflow.org/api_docs/python/tf/nn/bias_add) for more information.

**conv2d**

See the TensorFlow [conv2d documentation](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d) for more information.

**expand_dims**

See the TensorFlow [expand_dims documentation](https://www.tensorflow.org/api_docs/python/tf/expand_dims) for more information.

**gather**

See the TensorFlow [gather documentation](https://www.tensorflow.org/api_docs/python/tf/gather) for more information.

**identity**

See the TensorFlow [identity documentation](https://www.tensorflow.org/api_docs/python/tf/identity) for more information.

**local_response_normalization**

See the TensorFlow [local_response_normalization documentation](https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization)  for more information.

**max_pool**

See the TensorFlow [max_pool documentation](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool) for more information.

**placeholder**

See the TensorFlow [placeholder documentation](https://www.tensorflow.org/api_docs/python/tf/placeholder) for more information.

**reduce_mean**

See the TensorFlow [reduce_mean documentation](https://www.tensorflow.org/api_docs/python/tf/reduce_mean) for more information.

**relu**

See the TensorFlow [relu documentation](https://www.tensorflow.org/api_docs/python/tf/nn/relu) for more information.

**relu6**

See the TensorFlow [relu6 documentation](https://www.tensorflow.org/api_docs/python/tf/nn/relu6) for more information.

**rsqrt**

See the TensorFlow [rsqrt documentation](https://www.tensorflow.org/api_docs/python/tf/math/rsqrt) for more information.

**shape**

See the TensorFlow [shape documentation](https://www.tensorflow.org/api_docs/python/tf/shape) for more information.

**sigmoid**

See the TensorFlow [sigmoid documentation](https://www.tensorflow.org/api_docs/python/tf/sigmoid) for more information.

**softplus**

See the TensorFlow [softplus documentation](https://www.tensorflow.org/api_docs/python/tf/nn/softplus) for more information.

**squeeze**

See the TensorFlow [squeeze documentation](https://www.tensorflow.org/api_docs/python/tf/squeeze) for more information.

**tanh**

See the TensorFlow [tanh documentation](https://www.tensorflow.org/api_docs/python/tf/tanh) for more information.

**transpose**

See the TensorFlow [transpose documentation](https://www.tensorflow.org/api_docs/python/tf/transpose) for more information.

## Partially supported

**add**

The parser does not support all forms of [broadcast composition](https://www.tensorflow.org/performance/xla/broadcasting), only broadcasting of scalars and 1D tensors. See the TensorFlow [add operator documentation](https://www.tensorflow.org/api_docs/python/tf/add) for more information.

**add_n**

The parser does not support all forms of [broadcast composition](https://www.tensorflow.org/performance/xla/broadcasting), only broadcasting of scalars and 1D tensors. See the TensorFlow [add operator documentation](https://www.tensorflow.org/api_docs/python/tf/add_n) for more information.

**concat**

Arm NN supports concatenation along the channel dimension for data formats NHWC and NCHW.

**constant**

The parser does not support the optional `shape` argument. It always infers the shape of the output tensor from `value`. See the TensorFlow [constant documentation](https://www.tensorflow.org/api_docs/python/tf/constant) for further information.

**depthwise_conv2d_native**

The parser only supports a dilation rate of (1,1,1,1). See the TensorFlow [depthwise_conv2d_native documentation](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d_native) for more information.

**equal**

The parser does not support all forms of [broadcast composition](https://www.tensorflow.org/performance/xla/broadcasting), only broadcasting of 4D and 1D tensors. See the TensorFlow [equal operator documentation](https://www.tensorflow.org/api_docs/python/tf/math/equal) for more information.

**fused_batch_norm**

The parser does not support training outputs. See the TensorFlow [fused_batch_norm documentation](https://www.tensorflow.org/api_docs/python/tf/nn/fused_batch_norm) for more information.

**greater**

The parser does not support all forms of [broadcast composition](https://www.tensorflow.org/performance/xla/broadcasting), only broadcasting of 4D and 1D tensors. See the TensorFlow [greater operator documentation](https://www.tensorflow.org/api_docs/python/tf/math/greater) for more information.

**matmul**

The parser only supports constant weights in a fully connected layer. See the TensorFlow [matmul documentation](https://www.tensorflow.org/api_docs/python/tf/matmul) for more information.

**maximum**

where maximum is used in one of the following ways

* max(mul(a, x), x)
* max(mul(x, a), x)
* max(x, mul(a, x))
* max(x, mul(x, a)

This is interpreted as a ActivationLayer with a LeakyRelu activation function. Any other usage of max will result in the insertion of a simple maximum layer. The parser does not support all forms of [broadcast composition](https://www.tensorflow.org/performance/xla/broadcasting). See the TensorFlow [maximum documentation](https://www.tensorflow.org/api_docs/python/tf/maximum) for more information.

**minimum**

The parser does not support all forms of [broadcast composition](https://www.tensorflow.org/performance/xla/broadcasting), only broadcasting of 4D and 1D tensors. See the TensorFlow [minimum operator documentation](https://www.tensorflow.org/api_docs/python/tf/math/minimum) for more information.

**multiply**

The parser does not support all forms of [broadcast composition](https://www.tensorflow.org/performance/xla/broadcasting), only broadcasting of scalars and 1D tensors. See the TensorFlow [multiply documentation](https://www.tensorflow.org/api_docs/python/tf/multiply) for more information.

**pad**

Only supports tf.pad function with mode = 'CONSTANT' and constant_values = 0. See the TensorFlow [pad documentation](https://www.tensorflow.org/api_docs/python/tf/pad) for more information.

**realdiv**

The parser does not support all forms of [broadcast composition](https://www.tensorflow.org/performance/xla/broadcasting), only broadcasting of scalars and 1D tensors. See the TensorFlow [realdiv documentation](https://www.tensorflow.org/api_docs/python/tf/realdiv) for more information.

**reshape**

The parser does not support reshaping to or from 4D. See the TensorFlow [reshape documentation](https://www.tensorflow.org/api_docs/python/tf/reshape) for more information.

**resize_images**

The parser only supports `ResizeMethod.BILINEAR` with `align_corners=False`. See the TensorFlow [resize_images documentation](https://www.tensorflow.org/api_docs/python/tf/image/resize_images) for more information.

**softmax**

The parser only supports 2D inputs and does not support selecting the `softmax` dimension. See the TensorFlow [softmax documentation](https://www.tensorflow.org/api_docs/python/tf/nn/softmax) for more information.

**split**

Arm NN supports split along the channel dimension for data formats NHWC and NCHW.

**subtract**

The parser does not support all forms of broadcasting [broadcast composition](https://www.tensorflow.org/performance/xla/broadcasting), only broadcasting of scalars and 1D tensors. See the TensorFlow [subtract documentation](https://www.tensorflow.org/api_docs/python/tf/math/subtract) for more information.

**pack/stack**

See the TensorFlow [stack documentation](https://www.tensorflow.org/api_docs/python/tf/stack) for more information.

**strided_slice**

See the TensorFlow [strided_slice documentation](https://www.tensorflow.org/api_docs/python/tf/strided_slice) for more information.

## Tested networks

Arm tests these operators with the following TensorFlow fp32 neural networks:

* Cifar10

* Lenet

* Simple MNIST. For more information check out the [tutorial](https://developer.arm.com/technologies/machine-learning-on-arm/developer-material/how-to-guides/deploying-a-tensorflow-mnist-model-on-arm-nn) on the Arm Developer portal.

* mobilenet_v1_1.0_224. The Arm NN SDK only supports the non-quantized version of the network. See the [MobileNet_v1 documentation](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) for more information on quantized networks.

* inception_v3. The Arm NN SDK only supports the official inception_v3 transformed model. See the TensorFlow documentation on [preparing models for mobile deployment](https://www.tensorflow.org/mobile/prepare_models) for more information on how to transform the inception_v3 network.

* ResNet v2 50 implementation from the [TF Slim model zoo](https://github.com/tensorflow/models/tree/master/research/slim)

More machine learning operators will be supported in future releases.
