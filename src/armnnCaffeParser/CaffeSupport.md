#Caffe layers supported by the Arm NN SDK
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

The Arm NN SDK supports the following machine learning layers for Caffe networks: 


- BatchNorm, in inference mode. 
- Convolution, excluding the Dilation Size, Weight Filler, Bias Filler, Engine, Force nd_im2col, and Axis parameters.
- Eltwise, excluding the coeff parameter.
- Inner Product, excluding the Weight Filler, Bias Filler, Engine, and Axis parameters.
- Input.
- LRN, excluding the Engine parameter.
- Pooling, excluding the Stochastic Pooling and Engine parameters.
- ReLU.
- Scale.
- Softmax, excluding the Axis and Engine parameters.
- Split.
- Dropout, in inference mode.

More machine learning layers will be supported in future releases. 