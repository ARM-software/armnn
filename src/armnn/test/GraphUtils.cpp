//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "GraphUtils.hpp"

bool GraphHasNamedLayer(const armnn::Graph& graph, const std::string& name)
{
    for (auto&& layer : graph)
    {
        if (layer->GetName() == name)
        {
            return true;
        }
    }
    return false;
}

armnn::Layer* GetFirstLayerWithName(armnn::Graph& graph, const std::string& name)
{
    for (auto&& layer : graph)
    {
        if (layer->GetNameStr() == name)
        {
            return layer;
        }
    }
    return nullptr;
}

bool CheckNumberOfInputSlot(armnn::Layer* layer, unsigned int num)
{
    return layer->GetNumInputSlots() == num;
}

bool CheckNumberOfOutputSlot(armnn::Layer* layer, unsigned int num)
{
    return layer->GetNumOutputSlots() == num;
}

bool IsConnected(armnn::Layer* srcLayer, armnn::Layer* destLayer,
                 unsigned int srcSlot, unsigned int destSlot,
                 const armnn::TensorInfo& expectedTensorInfo)
{
    const armnn::IOutputSlot& outputSlot = srcLayer->GetOutputSlot(srcSlot);
    const armnn::TensorInfo& tensorInfo = outputSlot.GetTensorInfo();
    if (expectedTensorInfo != tensorInfo)
    {
        return false;
    }
    const unsigned int numConnections = outputSlot.GetNumConnections();
    for (unsigned int c = 0; c < numConnections; ++c)
    {
        auto inputSlot = boost::polymorphic_downcast<const armnn::InputSlot*>(outputSlot.GetConnection(c));
        if (inputSlot->GetOwningLayer().GetNameStr() == destLayer->GetNameStr() &&
            inputSlot->GetSlotIndex() == destSlot)
        {
            return true;
        }
    }
    return false;
}
