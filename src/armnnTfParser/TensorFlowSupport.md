#TensorFlow operators that the Arm NN SDK supports 

This reference guide provides a list of TensorFlow operators the Arm NN SDK currently supports. 

The Arm NN SDK TensorFlow parser currently only supports fp32 operators. 

These are the TensorFlow operators that the Arm NN SDK currently supports:    

**avg_pool** 

See the TensorFlow [avg_pool documentation](https://www.tensorflow.org/api_docs/python/tf/nn/avg_pool) for more information. 

**bias_add**

 See the TensorFlow [bias_add documentation](https://www.tensorflow.org/api_docs/python/tf/nn/bias_add) for more information. 

**conv2d** 

 See the TensorFlow [conv2d documentation](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d) for more information. 

**identity** 

See the TensorFlow [identity documentation](https://www.tensorflow.org/api_docs/python/tf/identity) for more information. 

**local_response_normalization** 

See the TensorFlow [local_response_normalization documentation](https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization)  for more information. 

**max_pool**  

See the TensorFlow [max_pool documentation](https://www.tensorflow.org/api_docs/python/tf/nn/max_pool) for more information. 

**relu** 

 See the TensorFlow [relu documentation](https://www.tensorflow.org/api_docs/python/tf/nn/relu) for more information. 

**relu6** 

 See the TensorFlow [relu6 documentation](https://www.tensorflow.org/api_docs/python/tf/nn/relu6) for more information. 

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

The Arm NN SDK TensorFlow parser currently partially supports: 

**add** 

The parser does not support all forms of [broadcast composition](https://www.tensorflow.org/performance/xla/broadcasting), only broadcasting of scalars and 1D tensors. See the TensorFlow [add operator documentation](https://www.tensorflow.org/api_docs/python/tf/add) for more information. 

**depthwise_conv2D_native** 

The parser only supports a dilation rate of (1,1,1,1). See the TensorFlow [depthwise_conv2d_native documentation](https://www.tensorflow.org/api_docs/python/tf/nn/depthwise_conv2d_native) for more information. 

**fused_batch_norm** 

The parser does not support training outputs. See the TensorFlow [fused_batch_norm documentation](https://www.tensorflow.org/api_docs/python/tf/nn/fused_batch_norm) for more information. 

**matmul** 

The parser only supports constant weights in a fully connected layer. See the TensorFlow [matmul documentation](https://www.tensorflow.org/api_docs/python/tf/matmul) for more information.  

**multiply** 

The parser does not support all forms of [broadcast composition](https://www.tensorflow.org/performance/xla/broadcasting), only broadcasting of scalars and 1D tensors. See the TensorFlow [multiply documentation](https://www.tensorflow.org/api_docs/python/tf/multiply) for more information. No broadcasting supported on the NEON backend.

**placeholder** 

 The parser only supports the NHWC data format in the input layer. See the TensorFlow [placeholder documentation](https://www.tensorflow.org/api_docs/python/tf/placeholder) for more information. 

**reshape** 

The parser does not support reshaping to or from 4D. See the TensorFlow [reshape documentation](https://www.tensorflow.org/api_docs/python/tf/reshape) for more information. 

**resize_images** 

The parser only supports `ResizeMethod.BILINEAR`. See the TensorFlow [resize_images documentation](https://www.tensorflow.org/api_docs/python/tf/image/resize_images) for more information. 
 
**softmax** 

The parser only supports 2D inputs and does not support selecting the `softmax` dimension. See the TensorFlow [softmax documentation](https://www.tensorflow.org/api_docs/python/tf/nn/softmax) for more information. 

 

Arm tests these operators with the following TensorFlow fp32 neural networks:  

* Cifar10. 

* Lenet. 

* mobilenet_v1_1.0_224. The Arm NN SDK only supports the non*_quant version of the network. See the [MobileNet_v1 documentation](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md) for more information on _quant networks. 

* inception_v3. The Arm NN SDK only supports the official inception_v3 transformed model using the GPU acceleration only, but NEON acceleration is not supported at the moment. See the TensorFlow documentation on [preparing models for mobile deployment](https://www.tensorflow.org/mobile/prepare_models) for more information on how to transform the inception_v3 network.

More machine learning operators will be supported in future releases. 
