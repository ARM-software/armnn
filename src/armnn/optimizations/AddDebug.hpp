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
            InsertDebugLayerAfter(graph, layer, false);
        }
    }

protected:
    AddDebugImpl() = default;
    ~AddDebugImpl() = default;
};

class AddDebugToFileImpl
{
public:

    void Run(Graph& graph, Layer& layer) const
    {
        if (layer.GetType() != LayerType::Debug && layer.GetType() != LayerType::Output)
        {
            // if the inputs/outputs of this layer do not have a debug layer
            // insert the debug layer after them
            InsertDebugLayerAfter(graph, layer, true);
        }
    }

protected:
    AddDebugToFileImpl() = default;
    ~AddDebugToFileImpl() = default;
};

using InsertDebugLayer = OptimizeForType<Layer, AddDebugImpl>;
using InsertDebugToFileLayer = OptimizeForType<Layer, AddDebugToFileImpl>;

} // namespace optimizations
} // namespace armnn
