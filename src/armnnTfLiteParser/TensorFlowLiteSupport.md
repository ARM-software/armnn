# TensorFlow Lite operators that the Arm NN SDK supports

This reference guide provides a list of TensorFlow Lite operators the Arm NN SDK currently supports.

The Arm NN SDK TensorFlow Lite parser currently only supports uint8.

## Fully supported

The Arm NN SDK TensorFlow Lite parser currently supports the following operators:

* AVERAGE_POOL_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* CONV_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* DEPTHWISE_CONV_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* SOFTMAX

* SQUEEZE

## Tested networks

Arm tested these operators with the following TensorFlow Lite neural network:

* [Quantized MobileNet](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224_quant.tgz)

More machine learning operators will be supported in future releases.