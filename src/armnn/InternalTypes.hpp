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
    BatchNormalization,
    BatchToSpaceNd,
    Constant,
    ConvertFp16ToFp32,
    ConvertFp32ToFp16,
    Convolution2d,
    Debug,
    DepthwiseConvolution2d,
    Division,
    FakeQuantization,
    Floor,
    FullyConnected,
    Input,
    L2Normalization,
    Lstm,
    Maximum,
    Mean,
    MemCopy,
    Merger,
    Minimum,
    Multiplication,
    Normalization,
    Output,
    Pad,
    Permute,
    Pooling2d,
    Reshape,
    ResizeBilinear,
    Softmax,
    SpaceToBatchNd,
    Splitter,
    StridedSlice,
    // Last layer goes here.
    LastLayer,
    Subtraction = LastLayer,
};

const char* GetLayerTypeAsCString(LayerType type);

using Coordinates = std::array<unsigned int, MaxNumOfTensorDimensions>;
using Dimensions = std::array<unsigned int, MaxNumOfTensorDimensions>;

}
