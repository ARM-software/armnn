# ONNX operators that the Arm NN SDK supports

This reference guide provides a list of ONNX operators the Arm NN SDK currently supports.

The Arm NN SDK ONNX parser currently only supports fp32 operators.

## Fully supported

**Add**

See the ONNX [Add documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Add) for more information

**AveragePool**

See the ONNX [AveragePool documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#AveragePool) for more information.

**Constant**

See the ONNX [Constant documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Constant) for more information.

**Clip**

See the ONNX [Clip documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Clip) for more information.

**Flatten**

See the ONNX [Flatten documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Flatten) for more information.

**GlobalAveragePool**

See the ONNX [GlobalAveragePool documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#GlobalAveragePool) for more information.

**LeakyRelu**

See the ONNX [LeakyRelu documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#LeakyRelu) for more information.


**MaxPool**

See the ONNX [max_pool documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#MaxPool) for more information.

**Relu**

See the ONNX [Relu documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Relu) for more information.

**Reshape**

See the ONNX [Reshape documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Reshape) for more information.

**Sigmoid**

See the ONNX [Sigmoid documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Sigmoid) for more information.


**Tanh**

See the ONNX [Tanh documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Tanh) for more information.


## Partially supported

**Conv**

The parser only supports 2D convolutions with a dilation rate of [1, 1] and group = 1 or group = #Nb_of_channel (depthwise convolution)
See the ONNX [Conv documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#Conv) for more information.

**BatchNormalization**

The parser does not support training mode. See the ONNX [BatchNormalization documentation](https://github.com/onnx/onnx/blob/master/docs/Operators.md#BatchNormalization) for more information.

**MatMul**

The parser only supports constant weights in a fully connected layer.

## Tested networks

Arm tested these operators with the following ONNX fp32 neural networks:

* Mobilenet_v2. See the ONNX [MobileNet documentation](https://github.com/onnx/models/tree/master/vision/classification/mobilenet) for more information.

* Simple MNIST. This is no longer directly documented by ONNX. The model and test data may be downloaded [from the ONNX model zoo](https://onnxzoo.blob.core.windows.net/models/opset_8/mnist/mnist.tar.gz).

More machine learning operators will be supported in future releases as time allows. If you require specific operator support contribution are welcome.
