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

* DIV

* EXP

* FULLY_CONNECTED, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* LEAKY_RELU

* LOGISTIC

* L2_NORMALIZATION

* MAX_POOL_2D, Supported Fused Activation: RELU , RELU6 , TANH, NONE

* MAXIMUM

* MEAN

* MINIMUM

* MUL

* NEG

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

* SPLIT_V

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

* [DeepLab v3+](https://www.tensorflow.org/lite/models/segmentation/overview)

* FSRCNN

* EfficientNet-lite

* RDN converted from [TensorFlow model](https://github.com/hengchuan/RDN-TensorFlow)

* Quantized RDN (CpuRef)

* [Quantized Inception v3](http://download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz)

* [Quantized Inception v4](http://download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz) (CpuRef)

* Quantized ResNet v2 50 (CpuRef)

* Quantized Yolo v3 (CpuRef)

More machine learning operators will be supported in future releases.
