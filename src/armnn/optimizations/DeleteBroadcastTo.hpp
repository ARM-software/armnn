//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Optimization.hpp"

namespace armnn
{
namespace optimizations
{
class DeleteBroadcastToImpl
{
public:
    /// Run for every BroadcastToLayer. Remove it if it is before an ElementWiseLayer.
    /// Since ElementWiseBinary uses a brodcastLoop, using a broadcastTo layer is
    /// not necessary so it will be deleted.
    void Run(Graph&, BroadcastToLayer& layer) const
    {
        if(layer.GetType() == LayerType::BroadcastTo)
        {
            TensorInfo info = layer.GetOutputSlot(0).GetTensorInfo();
            Layer& next = layer.GetOutputSlot(0).GetConnection(0)->GetOwningLayer();
            if (next.GetType() == LayerType::ElementwiseBinary)
            {
                Layer& connectedLayer = layer.GetInputSlots()[0].GetConnectedOutputSlot()->GetOwningLayer();
                auto tensorInfo = connectedLayer.GetOutputSlot().GetTensorInfo();
                layer.GetOutputSlot().MoveAllConnections(connectedLayer.GetOutputSlot());
                connectedLayer.GetOutputSlot().GetOutputHandler().SetTensorInfo(tensorInfo);
            }
        }
    }
protected:
    DeleteBroadcastToImpl() = default;
    ~DeleteBroadcastToImpl() = default;
};
using BroadcastToOptimizationLayer = OptimizeForType<BroadcastToLayer, DeleteBroadcastToImpl>;
}
}