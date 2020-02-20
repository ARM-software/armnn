# TensorFlow Lite operators that the Arm NN SDK supports

This reference guide provides a list of TensorFlow Lite operators the Arm NN SDK currently supports.

## Fully supported

The Arm NN SDK TensorFlow Lite parser currently supports the following operators:

* ADD

* AVERAGE_POOL_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* BATCH_TO_SPACE

* CONCATENATION, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* CONV_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* DEPTHWISE_CONV_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* DEQUANTIZE

* FULLY_CONNECTED, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* LOGISTIC

* L2_NORMALIZATION

* MAX_POOL_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* MAXIMUM

* MEAN

* MINIMUM

* MUL

* PACK

* PAD

* QUANTIZE

* RELU

* RELU6

* RESHAPE

* RESIZE_BILINEAR

* RESIZE_NEAREST_NEIGHBOR

* SLICE

* SOFTMAX

* SPACE_TO_BATCH

* SPLIT

* SQUEEZE

* STRIDED_SLICE

* SUB

* TANH

* TRANSPOSE

* TRANSPOSE_CONV

* UNPACK

## Custom Operator

* TFLite_Detection_PostProcess

## Tested networks

Arm tested these operators with the following TensorFlow Lite neural network:

* [Quantized MobileNet](http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224_quant.tgz)

* [Quantized SSD MobileNet](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_quantized_300x300_coco14_sync_2018_07_18.tar.gz)

* DeepSpeech v1 converted from [TensorFlow model](https://github.com/mozilla/DeepSpeech/releases/tag/v0.4.1)

* DeepSpeaker

More machine learning operators will be supported in future releases.
