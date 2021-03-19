//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Optimization.hpp"

#include <armnn/utility/PolymorphicDowncast.hpp>

namespace armnn
{
namespace optimizations
{
namespace
{
template <typename Layer2dT>
Layer2dT* FoldPadIntoLayer2dImpl(Graph& graph, InputSlot& connection)
{
    Layer& base = connection.GetConnectedOutputSlot()->GetOwningLayer();
    Layer& child = connection.GetOwningLayer();

    ARMNN_ASSERT(base.GetType() == LayerType::Pad);
    ARMNN_ASSERT(child.GetType() == LayerEnumOf<Layer2dT>());

    PadLayer* padLayer = PolymorphicDowncast<PadLayer*>(&base);
    Layer2dT* layer2d = PolymorphicDowncast<Layer2dT*>(&child);

    OutputSlot* parentOut = base.GetInputSlot(0).GetConnectedOutputSlot();

    const std::string name = std::string("folded-") + base.GetName() + std::string("-into-") + child.GetName();
    auto descriptor = layer2d->GetParameters();

    auto padList = padLayer->GetParameters().m_PadList;

    armnn::DataLayout dataLayout = descriptor.m_DataLayout;

    // In Convolution2dDescriptor/Pooling2dDescriptor, padLeft and padRight are defined as paddings
    // on width dimension whereas padTop and padBottom - paddings on height dimension, so setting these
    // according to data layout
    if(dataLayout == armnn::DataLayout::NHWC)
    {
        descriptor.m_PadLeft = padList[2].first;
        descriptor.m_PadRight = padList[2].second;
        descriptor.m_PadTop = padList[1].first;
        descriptor.m_PadBottom = padList[1].second;
    }
    else
    {
        descriptor.m_PadLeft = padList[3].first;
        descriptor.m_PadRight = padList[3].second;
        descriptor.m_PadTop = padList[2].first;
        descriptor.m_PadBottom = padList[2].second;
    }

    const auto newLayer2d = graph.InsertNewLayer<Layer2dT>(base.GetInputSlot(0), descriptor, name.c_str());

    // Reconnects with original parent.
    newLayer2d->GetOutputSlot().MoveAllConnections(*parentOut);
    // Parent is now the new layer.
    parentOut = &newLayer2d->GetOutputSlot();

    // Moves connections in child output to parent layer.
    // Child layer will be removed as it's left unconnected.
    // Base layer will be removed if left unconnected.
    child.GetOutputSlot().MoveAllConnections(*parentOut);

    return newLayer2d;
}
} // namespace

class FoldPadIntoConvolution2dImpl
{
public:
    void Run(Graph& graph, InputSlot& connection) const
    {
        const auto conv2dLayer = PolymorphicDowncast<Convolution2dLayer*>(&connection.GetOwningLayer());
        const auto newConv2dLayer = FoldPadIntoLayer2dImpl<Convolution2dLayer>(graph, connection);

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

protected:
    FoldPadIntoConvolution2dImpl() =  default;
    ~FoldPadIntoConvolution2dImpl() = default;
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

using FoldPadIntoConvolution2d = OptimizeForConnection<PadLayer, Convolution2dLayer, FoldPadIntoConvolution2dImpl>;
using FoldPadIntoPooling2d = OptimizeForConnection<PadLayer, Pooling2dLayer, FoldPadIntoPooling2dImpl>;

} // namespace optimizations
} // namespace armnn


