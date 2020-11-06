//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>

#include <array>


/// This list uses X macro technique.
/// See https://en.wikipedia.org/wiki/X_Macro for more info
#define LIST_OF_LAYER_TYPE \
    X(Activation) \
    X(Addition) \
    X(ArgMinMax) \
    X(BatchNormalization) \
    X(BatchToSpaceNd) \
    X(Comparison) \
    X(Concat) \
    X(Constant) \
    X(ConvertBf16ToFp32) \
    X(ConvertFp16ToFp32) \
    X(ConvertFp32ToBf16) \
    X(ConvertFp32ToFp16) \
    X(Convolution2d) \
    X(Debug) \
    X(DepthToSpace) \
    X(DepthwiseConvolution2d) \
    X(Dequantize) \
    X(DetectionPostProcess) \
    X(Division) \
    X(ElementwiseUnary) \
    X(FakeQuantization) \
    X(Fill) \
    X(Floor) \
    X(FullyConnected) \
    X(Gather) \
    X(Input) \
    X(InstanceNormalization) \
    X(L2Normalization) \
    X(LogicalBinary) \
    X(LogSoftmax) \
    X(Lstm) \
    X(QLstm) \
    X(Map) \
    X(Maximum) \
    X(Mean) \
    X(MemCopy) \
    X(MemImport) \
    X(Merge) \
    X(Minimum) \
    X(Multiplication) \
    X(Normalization) \
    X(Output) \
    X(Pad) \
    X(Permute) \
    X(Pooling2d) \
    X(PreCompiled) \
    X(Prelu) \
    X(Quantize) \
    X(QuantizedLstm) \
    X(Reshape) \
    X(Rank) \
    X(Resize) \
    X(Slice) \
    X(Softmax) \
    X(SpaceToBatchNd) \
    X(SpaceToDepth) \
    X(Splitter) \
    X(Stack) \
    X(StandIn) \
    X(StridedSlice) \
    X(Subtraction) \
    X(Switch) \
    X(Transpose) \
    X(TransposeConvolution2d) \
    X(Unmap)

/// When adding a new layer, adapt also the LastLayer enum value in the
/// enum class LayerType below
namespace armnn
{

enum class LayerType
{
#define X(name) name,
  LIST_OF_LAYER_TYPE
#undef X
  FirstLayer = Activation,
  LastLayer = Unmap
};

const char* GetLayerTypeAsCString(LayerType type);

using Coordinates = std::array<unsigned int, MaxNumOfTensorDimensions>;
using Dimensions  = std::array<unsigned int, MaxNumOfTensorDimensions>;

}
