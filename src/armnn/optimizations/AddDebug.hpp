//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "Optimization.hpp"
#include "NetworkUtils.hpp"

namespace armnn
{
namespace optimizations
{

class AddDebugImpl
{
public:

    void Run(Graph& graph, Layer& layer) const
    {
        if (layer.GetType() != LayerType::Debug && layer.GetType() != LayerType::Output)
        {
            // if the inputs/outputs of this layer do not have a debug layer
            // insert the debug layer after them
            InsertDebugLayerAfter(graph, layer);
        }
    }

protected:
    AddDebugImpl() = default;
    ~AddDebugImpl() = default;
};

using InsertDebugLayer = OptimizeForType<Layer, AddDebugImpl>;

} // namespace optimizations
} // namespace armnn
