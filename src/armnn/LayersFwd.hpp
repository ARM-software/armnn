//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "InternalTypes.hpp"

#include "layers/ActivationLayer.hpp"
#include "layers/AdditionLayer.hpp"
#include "layers/ArgMinMaxLayer.hpp"
#include "layers/BatchNormalizationLayer.hpp"
#include "layers/BatchToSpaceNdLayer.hpp"
#include "layers/CastLayer.hpp"
#include "layers/ChannelShuffleLayer.hpp"
#include "layers/ComparisonLayer.hpp"
#include "layers/ConcatLayer.hpp"
#include "layers/ConstantLayer.hpp"
#include "layers/ConvertBf16ToFp32Layer.hpp"
#include "layers/ConvertFp16ToFp32Layer.hpp"
#include "layers/ConvertFp32ToBf16Layer.hpp"
#include "layers/ConvertFp32ToFp16Layer.hpp"
#include "layers/Convolution2dLayer.hpp"
#include "layers/Convolution3dLayer.hpp"
#include "layers/DebugLayer.hpp"
#include "layers/DepthToSpaceLayer.hpp"
#include "layers/DepthwiseConvolution2dLayer.hpp"
#include "layers/DequantizeLayer.hpp"
#include "layers/DetectionPostProcessLayer.hpp"
#include "layers/DivisionLayer.hpp"
#include "layers/ElementwiseUnaryLayer.hpp"
#include "layers/FakeQuantizationLayer.hpp"
#include "layers/FillLayer.hpp"
#include "layers/FloorLayer.hpp"
#include "layers/FullyConnectedLayer.hpp"
#include "layers/GatherLayer.hpp"
#include "layers/InputLayer.hpp"
#include "layers/InstanceNormalizationLayer.hpp"
#include "layers/L2NormalizationLayer.hpp"
#include "layers/LogicalBinaryLayer.hpp"
#include "layers/LogSoftmaxLayer.hpp"
#include "layers/LstmLayer.hpp"
#include "layers/MapLayer.hpp"
#include "layers/MaximumLayer.hpp"
#include "layers/MeanLayer.hpp"
#include "layers/MemCopyLayer.hpp"
#include "layers/MemImportLayer.hpp"
#include "layers/MergeLayer.hpp"
#include "layers/MinimumLayer.hpp"
#include "layers/MultiplicationLayer.hpp"
#include "layers/NormalizationLayer.hpp"
#include "layers/OutputLayer.hpp"
#include "layers/PadLayer.hpp"
#include "layers/PermuteLayer.hpp"
#include "layers/Pooling2dLayer.hpp"
#include "layers/Pooling3dLayer.hpp"
#include "layers/PreCompiledLayer.hpp"
#include "layers/PreluLayer.hpp"
#include "layers/QuantizeLayer.hpp"
#include "layers/QLstmLayer.hpp"
#include "layers/QuantizedLstmLayer.hpp"
#include "layers/RankLayer.hpp"
#include "layers/ReduceLayer.hpp"
#include "layers/ReshapeLayer.hpp"
#include "layers/ResizeLayer.hpp"
#include "layers/ShapeLayer.hpp"
#include "layers/SliceLayer.hpp"
#include "layers/SoftmaxLayer.hpp"
#include "layers/SpaceToBatchNdLayer.hpp"
#include "layers/SpaceToDepthLayer.hpp"
#include "layers/SplitterLayer.hpp"
#include "layers/StackLayer.hpp"
#include "layers/StandInLayer.hpp"
#include "layers/StridedSliceLayer.hpp"
#include "layers/SubtractionLayer.hpp"
#include "layers/SwitchLayer.hpp"
#include "layers/TransposeConvolution2dLayer.hpp"
#include "layers/TransposeLayer.hpp"
#include "layers/UnidirectionalSequenceLstmLayer.hpp"
#include "layers/UnmapLayer.hpp"

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
DECLARE_LAYER(ArgMinMax)
DECLARE_LAYER(BatchNormalization)
DECLARE_LAYER(BatchToSpaceNd)
DECLARE_LAYER(Cast)
DECLARE_LAYER(ChannelShuffle)
DECLARE_LAYER(Comparison)
DECLARE_LAYER(Concat)
DECLARE_LAYER(Constant)
DECLARE_LAYER(ConvertBf16ToFp32)
DECLARE_LAYER(ConvertFp16ToFp32)
DECLARE_LAYER(ConvertFp32ToBf16)
DECLARE_LAYER(ConvertFp32ToFp16)
DECLARE_LAYER(Convolution2d)
DECLARE_LAYER(Convolution3d)
DECLARE_LAYER(Debug)
DECLARE_LAYER(DepthToSpace)
DECLARE_LAYER(DepthwiseConvolution2d)
DECLARE_LAYER(Dequantize)
DECLARE_LAYER(DetectionPostProcess)
DECLARE_LAYER(Division)
DECLARE_LAYER(ElementwiseUnary)
DECLARE_LAYER(FakeQuantization)
DECLARE_LAYER(Fill)
DECLARE_LAYER(Floor)
DECLARE_LAYER(FullyConnected)
DECLARE_LAYER(Gather)
DECLARE_LAYER(Input)
DECLARE_LAYER(InstanceNormalization)
DECLARE_LAYER(L2Normalization)
DECLARE_LAYER(LogicalBinary)
DECLARE_LAYER(LogSoftmax)
DECLARE_LAYER(Lstm)
DECLARE_LAYER(Map)
DECLARE_LAYER(Maximum)
DECLARE_LAYER(Mean)
DECLARE_LAYER(MemCopy)
DECLARE_LAYER(MemImport)
DECLARE_LAYER(Merge)
DECLARE_LAYER(Minimum)
DECLARE_LAYER(Multiplication)
DECLARE_LAYER(Normalization)
DECLARE_LAYER(Output)
DECLARE_LAYER(Pad)
DECLARE_LAYER(Permute)
DECLARE_LAYER(Pooling2d)
DECLARE_LAYER(Pooling3d)
DECLARE_LAYER(PreCompiled)
DECLARE_LAYER(Prelu)
DECLARE_LAYER(Quantize)
DECLARE_LAYER(QLstm)
DECLARE_LAYER(QuantizedLstm)
DECLARE_LAYER(Rank)
DECLARE_LAYER(Reduce)
DECLARE_LAYER(Reshape)
DECLARE_LAYER(Resize)
DECLARE_LAYER(Shape)
DECLARE_LAYER(Slice)
DECLARE_LAYER(Softmax)
DECLARE_LAYER(SpaceToBatchNd)
DECLARE_LAYER(SpaceToDepth)
DECLARE_LAYER(Splitter)
DECLARE_LAYER(Stack)
DECLARE_LAYER(StandIn)
DECLARE_LAYER(StridedSlice)
DECLARE_LAYER(Subtraction)
DECLARE_LAYER(Switch)
DECLARE_LAYER(Transpose)
DECLARE_LAYER(TransposeConvolution2d)
DECLARE_LAYER(UnidirectionalSequenceLstm)
DECLARE_LAYER(Unmap)
}
