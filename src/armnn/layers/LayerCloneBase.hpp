//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include <Layer.hpp>
#include <Graph.hpp>

namespace armnn
{

template <typename LayerType, typename ... Params>
LayerType* Layer::CloneBase(Graph& graph, Params&& ... params) const
{
    LayerType* const layer = graph.AddLayer<LayerType>(std::forward<Params>(params)...);

    layer->SetComputeDevice(m_ComputeDevice);
    layer->SetGuid(GetGuid());

    return layer;
}

} // namespace
