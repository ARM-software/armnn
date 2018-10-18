//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
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

    layer->SetBackendId(GetBackendId());
    layer->SetGuid(GetGuid());

    return layer;
}

} // namespace
