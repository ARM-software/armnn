# The layers that ArmNN SDK Serializer currently supports.

This reference guide provides a list of layers which can be serialized currently by the Arm NN SDK.

## Fully supported

The Arm NN SDK Serializer currently supports the following layers:

* Activation
* Addition
* ArgMinMax
* BatchToSpaceNd
* BatchNormalization
* Comparison
* Concat
* Constant
* Convolution2d
* DepthToSpace
* DepthwiseConvolution2d
* Dequantize
* DetectionPostProcess
* Division
* ElementwiseUnary
* Floor
* FullyConnected
* Gather
* Input
* InstanceNormalization
* L2Normalization
* LogSoftmax
* Lstm
* Maximum
* Mean
* Merge
* Minimum
* Multiplication
* Normalization
* Output
* Pad
* Permute
* Pooling2d
* Prelu
* Quantize
* QuantizedLstm
* Reshape
* Resize
* ResizeBilinear
* Slice
* Softmax
* SpaceToBatchNd
* SpaceToDepth
* Splitter
* Stack
* StandIn
* StridedSlice
* Subtraction
* Switch
* TransposeConvolution2d

More machine learning layers will be supported in future releases.

## Deprecated layers

Some layers have been deprecated and replaced by others layers. In order to maintain backward compatibility, serializations of these deprecated layers will deserialize to the layers that have replaced them, as follows:

* Equal will deserialize as Comparison
* Merger will deserialize as Concat
* Greater will deserialize as Comparison
* ResizeBilinear will deserialize as Resize
* Abs will deserialize as ElementwiseUnary
* Rsqrt will deserialize as ElementwiseUnary

