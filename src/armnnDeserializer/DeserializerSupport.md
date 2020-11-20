# The layers that ArmNN SDK Deserializer currently supports.

This reference guide provides a list of layers which can be deserialized currently by the Arm NN SDK.

## Fully supported

The Arm NN SDK Deserialize parser currently supports the following layers:

* Abs
* Activation
* Addition
* ArgMinMax
* BatchToSpaceNd
* BatchNormalization
* Concat
* Comparison
* Constant
* Convolution2d
* DepthToSpace
* DepthwiseConvolution2d
* Dequantize
* DetectionPostProcess
* Division
* ElementwiseUnary
* Fill
* Floor
* FullyConnected
* Gather
* Input
* InstanceNormalization
* L2Normalization
* Logical
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
* QLstm
* QuantizedLstm
* Rank
* Reshape
* Resize
* ResizeBilinear
* Rsqrt
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
