//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
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
    Constant,
    ConvertFp16ToFp32,
    ConvertFp32ToFp16,
    Convolution2d,
    DepthwiseConvolution2d,
    FakeQuantization,
    Floor,
    FullyConnected,
    Input,
    L2Normalization,
    Lstm,
    MemCopy,
    Merger,
    Multiplication,
    Normalization,
    Output,
    Permute,
    Pooling2d,
    Reshape,
    ResizeBilinear,
    Softmax,
    // Last layer goes here.
    LastLayer,
    Splitter = LastLayer,
};

const char* GetLayerTypeAsCString(LayerType type);

using Coordinates = std::array<unsigned int, MaxNumOfTensorDimensions>;
using Dimensions = std::array<unsigned int, MaxNumOfTensorDimensions>;

}
