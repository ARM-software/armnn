//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Optimization.hpp"

#include <armnnUtils/QuantizeHelper.hpp>

#include <armnn/utility/PolymorphicDowncast.hpp>
#include <armnnUtils/DataLayoutIndexed.hpp>

namespace armnn
{
namespace optimizations
{
namespace pad_fold
{
inline float GetZeroElement(const TensorInfo& tensorInfo)
{
    return static_cast<float>(tensorInfo.IsQuantized() ? tensorInfo.GetQuantizationOffset() : 0);
}

inline float GetLowestElement(const TensorInfo& tensorInfo)
{
    constexpr float negativeInfinity = -std::numeric_limits<float>::infinity();
    const float scale = tensorInfo.GetQuantizationScale();
    const int32_t offset = tensorInfo.GetQuantizationOffset();

    switch (tensorInfo.GetDataType())
    {
        case DataType::Float16:
            return armnnUtils::SelectiveQuantize<armnn::Half>(negativeInfinity, scale, offset);
        case DataType::Float32:
            return armnnUtils::SelectiveQuantize<float>(negativeInfinity, scale, offset);
        case DataType::QAsymmU8:
            return armnnUtils::SelectiveQuantize<uint8_t>(negativeInfinity, scale, offset);
        case DataType::QSymmS16:
            return armnnUtils::SelectiveQuantize<int16_t>(negativeInfinity, scale, offset);
        case DataType::QSymmS8:
            // Fall-through
        case DataType::QAsymmS8:
            return armnnUtils::SelectiveQuantize<int8_t>(negativeInfinity, scale, offset);
        case DataType::BFloat16:
            return armnnUtils::SelectiveQuantize<armnn::BFloat16>(negativeInfinity, scale, offset);
        default:
        {
            ARMNN_ASSERT_MSG(false, "Unsupported DataType");
            return NAN;
        }
    }
}

inline bool IsNeutralElement(const Convolution2dDescriptor&, const TensorInfo& tensorInfo, const float tensorValue)
{
    return tensorValue == GetZeroElement(tensorInfo);
}

inline bool IsNeutralElement(const DepthwiseConvolution2dDescriptor&,
                             const TensorInfo& tensorInfo,
                             const float tensorValue)
{
    return tensorValue == GetZeroElement(tensorInfo);
}

inline bool IsNeutralElement(
    const Pooling2dDescriptor& descriptor, const TensorInfo& tensorInfo, const float tensorValue)
{
    return (descriptor.m_PoolType == PoolingAlgorithm::Max)
        ? tensorValue <= GetLowestElement(tensorInfo)
        : tensorValue == GetZeroElement(tensorInfo);
}

template <typename Descriptor>
bool TryFoldPadIntoLayer2d(
    const PadDescriptor& padDescriptor, Descriptor& layerDescriptor, const TensorInfo& tensorInfo)
{
    armnnUtils::DataLayoutIndexed layout = armnnUtils::DataLayoutIndexed(layerDescriptor.m_DataLayout);
    constexpr unsigned int batchIndex = 0;

    constexpr auto noPad = std::make_pair(0U, 0U);

    if ((!IsNeutralElement(layerDescriptor, tensorInfo, padDescriptor.m_PadValue)) ||
        (padDescriptor.m_PadList[batchIndex] != noPad) || (padDescriptor.m_PadList[layout.GetChannelsIndex()] != noPad))
    {
        return false;
    }

    const auto& padList = padDescriptor.m_PadList;

    // In Convolution2dDescriptor/Pooling2dDescriptor, padLeft and padRight are defined as paddings
    // on width dimension whereas padTop and padBottom - paddings on height dimension, so updating
    // these according to data layout
    layerDescriptor.m_PadLeft += padList[layout.GetWidthIndex()].first;
    layerDescriptor.m_PadRight += padList[layout.GetWidthIndex()].second;
    layerDescriptor.m_PadTop += padList[layout.GetHeightIndex()].first;
    layerDescriptor.m_PadBottom += padList[layout.GetHeightIndex()].second;

    return true;
}

inline bool TryFoldPadIntoLayer2d(
    const PadDescriptor& padDescriptor, Pooling2dDescriptor& poolDescriptor, const TensorInfo& tensorInfo)
{
    const auto poolingPadValues = std::make_tuple(poolDescriptor.m_PadLeft, poolDescriptor.m_PadRight,
                                                  poolDescriptor.m_PadTop, poolDescriptor.m_PadBottom);
    bool poolHasPadding = false;
    if (poolingPadValues != std::make_tuple(0U, 0U, 0U, 0U))
    {
        poolHasPadding = true;
    }

    // We cannot fold Average or L2 pooling if there's is already padding and that padding method is Exclude.
    if (poolDescriptor.m_PoolType != PoolingAlgorithm::Max) // PoolingAlgorithm::Average or PoolingAlgorithm::L2
    {
        if ((poolHasPadding) && (poolDescriptor.m_PaddingMethod == PaddingMethod::Exclude))
        {
            return false;
        }
    }
    poolDescriptor.m_PaddingMethod = PaddingMethod::IgnoreValue;

    return TryFoldPadIntoLayer2d<Pooling2dDescriptor>(padDescriptor, poolDescriptor, tensorInfo);
}

template <typename Layer2dT>
Layer2dT* FoldPadIntoLayer2dImpl(Graph& graph, InputSlot& connection)
{
    PadLayer& padLayer = *PolymorphicDowncast<PadLayer*>(&connection.GetConnectedOutputSlot()->GetOwningLayer());
    Layer2dT& layer2d = *PolymorphicDowncast<Layer2dT*>(&connection.GetOwningLayer());

    const PadDescriptor& padDescriptor = padLayer.GetParameters();
    auto newLayer2dDescriptor = layer2d.GetParameters();

    if (!TryFoldPadIntoLayer2d(padDescriptor, newLayer2dDescriptor, padLayer.GetOutputSlot().GetTensorInfo()))
    {
        return nullptr;
    }

    // Save original parent output slot of the pad layer
    OutputSlot& parentSlot = *padLayer.GetInputSlot(0).GetConnectedOutputSlot();

    // Insert new layer2d layer between the pad layer an its parent layer.
    const std::string name = std::string("folded-") + padLayer.GetName() + "-into-" + layer2d.GetName();
    auto& newLayer2d = *graph.InsertNewLayer<Layer2dT>(padLayer.GetInputSlot(0), newLayer2dDescriptor, name.c_str());

    // Reconnect the pad layer with its original parent.
    newLayer2d.GetOutputSlot().MoveAllConnections(parentSlot);

    // Moves connections in old layer2d layer output to new layer.
    // Old layer2d layer will be removed as it's left unconnected.
    // Pad layer will be removed if left unconnected.
    layer2d.GetOutputSlot().MoveAllConnections(newLayer2d.GetOutputSlot());

    return &newLayer2d;
}

class FoldPadIntoConvolution2dImpl
{
public:
    void Run(Graph& graph, InputSlot& connection) const
    {
        const auto newConv2dLayer = FoldPadIntoLayer2dImpl<Convolution2dLayer>(graph, connection);

        if (newConv2dLayer != nullptr)
        {
            const auto conv2dLayer = PolymorphicDowncast<Convolution2dLayer*>(&connection.GetOwningLayer());
            // Copy weights and bias to the new convolution layer
            ARMNN_ASSERT_MSG(conv2dLayer->m_Weight != nullptr,
                             "FoldPadIntoConvolution2d: Weights data should not be null.");
            newConv2dLayer->m_Weight = std::move(conv2dLayer->m_Weight);

            if (conv2dLayer->GetParameters().m_BiasEnabled)
            {
                ARMNN_ASSERT_MSG(conv2dLayer->m_Bias != nullptr,
                                 "FoldPadIntoConvolution2d: Bias data should not be null if bias is enabled.");
                newConv2dLayer->m_Bias = std::move(conv2dLayer->m_Bias);
            }
        }
    }

protected:
    FoldPadIntoConvolution2dImpl() =  default;
    ~FoldPadIntoConvolution2dImpl() = default;
};

class FoldPadIntoDepthwiseConvolution2dImpl
{
public:
    void Run(Graph& graph, InputSlot& connection) const
    {
        const auto newConv2dLayer = FoldPadIntoLayer2dImpl<DepthwiseConvolution2dLayer>(graph, connection);

        if (newConv2dLayer != nullptr)
        {
            const auto conv2dLayer = PolymorphicDowncast<DepthwiseConvolution2dLayer*>(&connection.GetOwningLayer());
            // Copy weights and bias to the new convolution layer
            ARMNN_ASSERT_MSG(conv2dLayer->m_Weight != nullptr,
                             "FoldPadIntoDepthwiseConvolution2d: Weights data should not be null.");
            newConv2dLayer->m_Weight = std::move(conv2dLayer->m_Weight);

            if (conv2dLayer->GetParameters().m_BiasEnabled)
            {
                ARMNN_ASSERT_MSG(conv2dLayer->m_Bias != nullptr,
                                "FoldPadIntoDepthwiseConvolution2d: Bias data should not be null if bias is enabled.");
                newConv2dLayer->m_Bias = std::move(conv2dLayer->m_Bias);
            }
        }
    }

protected:
    FoldPadIntoDepthwiseConvolution2dImpl() =  default;
    ~FoldPadIntoDepthwiseConvolution2dImpl() = default;
};

class FoldPadIntoPooling2dImpl
{
public:
    void Run(Graph& graph, InputSlot& connection) const
    {
        FoldPadIntoLayer2dImpl<Pooling2dLayer>(graph, connection);
    }

protected:
    FoldPadIntoPooling2dImpl() =  default;
    ~FoldPadIntoPooling2dImpl() = default;
};
} // namespace pad_fold

using FoldPadIntoConvolution2d =
    OptimizeForExclusiveConnection<PadLayer, Convolution2dLayer, pad_fold::FoldPadIntoConvolution2dImpl>;
using FoldPadIntoDepthwiseConvolution2d =
    OptimizeForExclusiveConnection <PadLayer,
                                    DepthwiseConvolution2dLayer,
                                    pad_fold::FoldPadIntoDepthwiseConvolution2dImpl>;
using FoldPadIntoPooling2d =
    OptimizeForExclusiveConnection<PadLayer, Pooling2dLayer, pad_fold::FoldPadIntoPooling2dImpl>;

} // namespace optimizations
} // namespace armnn


