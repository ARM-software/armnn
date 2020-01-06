//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>

#include <array>

namespace armnn
{

enum class LayerType
{
    FirstLayer,
    Activation = FirstLayer,
    Addition,
    ArgMinMax,
    BatchNormalization,
    BatchToSpaceNd,
    Comparison,
    Concat,
    Constant,
    ConvertFp16ToFp32,
    ConvertFp32ToFp16,
    Convolution2d,
    Debug,
    DepthToSpace,
    DepthwiseConvolution2d,
    Dequantize,
    DetectionPostProcess,
    Division,
    ElementwiseUnary,
    FakeQuantization,
    Floor,
    FullyConnected,
    Gather,
    Input,
    InstanceNormalization,
    L2Normalization,
    LogSoftmax,
    Lstm,
    Maximum,
    Mean,
    MemCopy,
    MemImport,
    Merge,
    Minimum,
    Multiplication,
    Normalization,
    Output,
    Pad,
    Permute,
    Pooling2d,
    PreCompiled,
    Prelu,
    Quantize,
    QuantizedLstm,
    Reshape,
    Resize,
    Slice,
    Softmax,
    SpaceToBatchNd,
    SpaceToDepth,
    Splitter,
    Stack,
    StandIn,
    StridedSlice,
    Subtraction,
    Switch,
    // Last layer goes here.
    LastLayer,
    TransposeConvolution2d = LastLayer
};

const char* GetLayerTypeAsCString(LayerType type);

using Coordinates = std::array<unsigned int, MaxNumOfTensorDimensions>;
using Dimensions  = std::array<unsigned int, MaxNumOfTensorDimensions>;

}
