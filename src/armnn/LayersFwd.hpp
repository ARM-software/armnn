//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "InternalTypes.hpp"

namespace armnn
{

template <LayerType Type>
struct LayerTypeOfImpl;

template <LayerType Type>
using LayerTypeOf = typename LayerTypeOfImpl<Type>::Type;

template <typename T>
constexpr LayerType LayerEnumOf(const T* = nullptr);

#define DECLARE_LAYER_IMPL(_, LayerName)                     \
    class LayerName##Layer;                                  \
    template <>                                              \
    struct LayerTypeOfImpl<LayerType::_##LayerName>          \
    {                                                        \
        using Type = LayerName##Layer;                       \
    };                                                       \
    template <>                                              \
    constexpr LayerType LayerEnumOf(const LayerName##Layer*) \
    {                                                        \
        return LayerType::_##LayerName;                      \
    }

#define DECLARE_LAYER(LayerName) DECLARE_LAYER_IMPL(, LayerName)

DECLARE_LAYER(Activation)
DECLARE_LAYER(Addition)
DECLARE_LAYER(BatchNormalization)
DECLARE_LAYER(Constant)
DECLARE_LAYER(Convolution2d)
DECLARE_LAYER(DepthwiseConvolution2d)
DECLARE_LAYER(FakeQuantization)
DECLARE_LAYER(Floor)
DECLARE_LAYER(FullyConnected)
DECLARE_LAYER(Input)
DECLARE_LAYER(L2Normalization)
DECLARE_LAYER(MemCopy)
DECLARE_LAYER(Merger)
DECLARE_LAYER(Multiplication)
DECLARE_LAYER(Normalization)
DECLARE_LAYER(Output)
DECLARE_LAYER(Permute)
DECLARE_LAYER(Pooling2d)
DECLARE_LAYER(Reshape)
DECLARE_LAYER(ResizeBilinear)
DECLARE_LAYER(Softmax)
DECLARE_LAYER(Splitter)

}
