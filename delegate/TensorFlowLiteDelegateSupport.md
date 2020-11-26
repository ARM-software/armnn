# TensorFlow Lite operators that the Arm NN TensorFlow Lite Delegate supports

This reference guide provides a list of TensorFlow Lite operators the Arm NN SDK currently supports.

## Fully supported

The Arm NN SDK TensorFlow Lite delegate currently supports the following operators:

* ABS

* ADD

* AVERAGE_POOL_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* CONCATENATION, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* CONV_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* DEPTHWISE_CONV_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* DEQUANTIZE

* DIV

* EQUAL

* EXP

* FULLY_CONNECTED

* GREATER

* GREATER_OR_EQUAL

* LESS

* LESS_OR_EQUAL

* LOGISTIC

* LOG_SOFTMAX

* L2_POOL_2D

* MAXIMUM

* MAX_POOL_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* MEAN

* MINIMUM

* MUL

* NEG

* NOT_EQUAL

* QUANTIZE

* RESHAPE

* RESIZE_BILINEAR

* RESIZE_NEAREST_NEIGHBOR

* RELU

* RELU6

* RSQRT

* SOFTMAX

* SQRT

* SUB

* TANH

* TRANSPOSE

* TRANSPOSE_CONV

More machine learning operators will be supported in future releases.
