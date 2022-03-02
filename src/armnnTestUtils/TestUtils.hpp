//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/INetwork.hpp>
#include <Graph.hpp>
#include <Runtime.hpp>

void Connect(armnn::IConnectableLayer* from, armnn::IConnectableLayer* to, const armnn::TensorInfo& tensorInfo,
             unsigned int fromIndex = 0, unsigned int toIndex = 0);

template <typename LayerT>
bool IsLayerOfType(const armnn::Layer* const layer)
{
    return (layer->GetType() == armnn::LayerEnumOf<LayerT>());
}

inline bool CheckSequence(const armnn::Graph::ConstIterator first, const armnn::Graph::ConstIterator last)
{
    return (first == last);
}

/// Checks each unary function in Us evaluates true for each correspondent layer in the sequence [first, last).
template <typename U, typename... Us>
bool CheckSequence(const armnn::Graph::ConstIterator first, const armnn::Graph::ConstIterator last, U&& u, Us&&... us)
{
    return u(*first) && CheckSequence(std::next(first), last, us...);
}

template <typename LayerT>
bool CheckRelatedLayers(armnn::Graph& graph, const std::list<std::string>& testRelatedLayers)
{
    for (auto& layer : graph)
    {
        if (layer->GetType() == armnn::LayerEnumOf<LayerT>())
        {
            auto& relatedLayers = layer->GetRelatedLayerNames();
            if (!std::equal(relatedLayers.begin(), relatedLayers.end(), testRelatedLayers.begin(),
                            testRelatedLayers.end()))
            {
                return false;
            }
        }
    }

    return true;
}

namespace armnn
{
Graph& GetGraphForTesting(IOptimizedNetwork* optNetPtr);
ModelOptions& GetModelOptionsForTesting(IOptimizedNetwork* optNetPtr);
arm::pipe::IProfilingService& GetProfilingService(RuntimeImpl* runtime);

} // namespace armnn
