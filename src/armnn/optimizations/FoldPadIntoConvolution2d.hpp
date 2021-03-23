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

class FoldPadIntoConvolution2dImpl
{
public:

    void Run(Graph& graph, InputSlot& connection) const
    {
        Layer& base = connection.GetConnectedOutputSlot()->GetOwningLayer();
        Layer& child = connection.GetOwningLayer();

        ARMNN_ASSERT(base.GetType() == LayerType::Pad);
        ARMNN_ASSERT(child.GetType() == LayerType::Convolution2d);

        PadLayer* padLayer = PolymorphicDowncast<PadLayer*>(&base);
        Convolution2dLayer* convolution2dLayer = PolymorphicDowncast<Convolution2dLayer*>(&child);

        OutputSlot* parentOut = base.GetInputSlot(0).GetConnectedOutputSlot();

        const std::string name = std::string("folded-") + base.GetName() + std::string("-into-") + child.GetName();
        Convolution2dDescriptor descriptor = convolution2dLayer->GetParameters();

        auto padList = padLayer->GetParameters().m_PadList;

        armnn::DataLayout dataLayout = descriptor.m_DataLayout;

        // In Convolution2dDescriptor, padLeft and padRight are defined as paddings on width dimension
        // whereas padTop and padBottom - paddings on height dimension, so setting these according to data layout
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

        auto& newConv2dLayer = *graph.InsertNewLayer<Convolution2dLayer>(base.GetInputSlot(0),
                                                                         descriptor,
                                                                         name.c_str());

        // Copy weights and bias to the new convolution layer
        ARMNN_ASSERT_MSG(convolution2dLayer->m_Weight != nullptr,
                         "FoldPadIntoConvolution2d: Weights data should not be null.");
        newConv2dLayer.m_Weight = std::move(convolution2dLayer->m_Weight);
        if (descriptor.m_BiasEnabled)
        {
            ARMNN_ASSERT_MSG(convolution2dLayer->m_Bias != nullptr,
                             "FoldPadIntoConvolution2d: Bias data should not be null if bias is enabled.");
            newConv2dLayer.m_Bias = std::move(convolution2dLayer->m_Bias);
        }

        // Reconnects with original parent.
        newConv2dLayer.GetOutputSlot().MoveAllConnections(*parentOut);
        // Parent is now the new convolution2d layer.
        parentOut = &newConv2dLayer.GetOutputSlot();

        // Moves connections in child output to parent layer.
        // Child layer will be removed as it's left unconnected.
        // Base layer will be removed if left unconnected.
        child.GetOutputSlot().MoveAllConnections(*parentOut);
    }
protected:
    FoldPadIntoConvolution2dImpl() =  default;
    ~FoldPadIntoConvolution2dImpl() = default;
};

using FoldPadIntoConvolution2d = OptimizeForConnection<PadLayer, Convolution2dLayer, FoldPadIntoConvolution2dImpl>;

} // namespace optimizations
} // namespace armnn


