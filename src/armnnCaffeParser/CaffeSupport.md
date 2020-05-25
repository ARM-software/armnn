# Caffe layers supported by the Arm NN SDK
This reference guide provides a list of Caffe layers the Arm NN SDK currently supports.

Although some other neural networks might work, Arm tests the Arm NN SDK with Caffe implementations of the following neural networks: 

- AlexNet.
- Cifar10.
- Inception-BN.
- Resnet_50, Resnet_101 and Resnet_152.
- VGG_CNN_S, VGG_16 and VGG_19.
- Yolov1_tiny.
- Lenet.
- MobileNetv1.
- SqueezeNet v1.0 and SqueezeNet v1.1

The Arm NN SDK supports the following machine learning layers for Caffe networks:


- BatchNorm, in inference mode.
- Convolution, excluding the Dilation Size, Weight Filler, Bias Filler, Engine, Force nd_im2col, and Axis parameters.

  Caffe doesn't support depthwise convolution, the equivalent layer is implemented through the notion of groups. ArmNN supports groups this way:
  - when group=1, it is a normal conv2d
  - when group=#input_channels, we can replace it by a depthwise convolution
  - when group>1 && group<#input_channels, we need to split the input into the given number of groups, apply a separate convolution and then merge the results
- Concat, along the channel dimension only.
- Dropout, in inference mode.
- Eltwise, excluding the coeff parameter.
- Inner Product, excluding the Weight Filler, Bias Filler, Engine, and Axis parameters.
- Input.
- LRN, excluding the Engine parameter.
- Pooling, excluding the Stochastic Pooling and Engine parameters.
- ReLU.
- Scale.
- Softmax, excluding the Axis and Engine parameters.
- Split.

More machine learning layers will be supported in future releases.