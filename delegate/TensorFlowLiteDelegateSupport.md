# TensorFlow Lite operators that the Arm NN TensorFlow Lite Delegate supports

This reference guide provides a list of TensorFlow Lite operators the Arm NN SDK currently supports.

## Fully supported

The Arm NN SDK TensorFlow Lite delegate currently supports the following operators:

* ABS

* ADD

* ARGMAX

* ARGMIN

* AVERAGE_POOL_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* CONCATENATION, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* CONV_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* DEPTHWISE_CONV_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* DEQUANTIZE

* DIV

* EQUAL

* ELU

* EXP

* FULLY_CONNECTED, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* GATHER

* GREATER

* GREATER_OR_EQUAL

* HARD_SWISH

* LESS

* LESS_OR_EQUAL

* LOCAL_RESPONSE_NORMALIZATION

* LOGICAL_AND
  
* LOGICAL_NOT
  
* LOGICAL_OR

* LOGISTIC

* LOG_SOFTMAX

* L2_NORMALIZATION

* L2_POOL_2D

* MAXIMUM

* MAX_POOL_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* MEAN

* MINIMUM

* MUL

* NEG

* NOT_EQUAL

* PAD

* QUANTIZE

* RESHAPE

* RESIZE_BILINEAR

* RESIZE_NEAREST_NEIGHBOR

* RELU

* RELU6

* RSQRT

* SOFTMAX

* SPLIT

* SPLIT_V

* SQRT

* SUB

* TANH

* TRANSPOSE

* TRANSPOSE_CONV

More machine learning operators will be supported in future releases.
