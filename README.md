# Arm NN

For more information about Arm NN, see: <https://developer.arm.com/products/processors/machine-learning/arm-nn>

There is a getting started guide here using TensorFlow: <https://developer.arm.com/technologies/machine-learning-on-arm/developer-material/how-to-guides/configuring-the-arm-nn-sdk-build-environment-for-tensorflow>

There is a getting started guide here using TensorFlow Lite: [TensorFlow Lite Support](src/armnnTfLiteParser/README.md)

There is a getting started guide here using Caffe: <https://developer.arm.com/technologies/machine-learning-on-arm/developer-material/how-to-guides/configuring-the-arm-nn-sdk-build-environment-for-caffe>

There is a getting started guide here using ONNX: [ONNX Support](src/armnnOnnxParser/README.md)

### Build Instructions

Arm tests the build system of Arm NN with the following build environments:

* Android NDK: [How to use Android NDK to build ArmNN](BuildGuideAndroidNDK.md)
* Cross compilation from x86_64 Ubuntu to arm64 Linux: [ArmNN Cross Compilation](BuildGuideCrossCompilation.md)
* Native compilation under arm64 Debian 9

Arm NN is written using portable C++14 and the build system uses [CMake](https://cmake.org/) so it is possible to build for a wide variety of target platforms, from a wide variety of host environments.

The armnn/tests directory contains tests used during ArmNN development. Many of them depend on third-party IP, model protobufs and image files not distributed with ArmNN. The dependencies of some of the tests are available freely on the Internet, for those who wish to experiment.

The 'ExecuteNetwork' program, in armnn/tests/ExecuteNetwork, has no additional dependencies beyond those required by ArmNN and the model parsers. It takes any model and any input tensor, and simply prints out the output tensor. Run with no arguments to see command-line help.

The 'armnn/samples' directory contains SimpleSample.cpp. A very basic example of the ArmNN SDK API in use.