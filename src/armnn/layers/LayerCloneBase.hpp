//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
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

    layer->BackendSelectionHint(GetBackendHint());
    layer->SetBackendId(GetBackendId());
    layer->SetGuid(GetGuid());
    layer->SetShapeInferenceMethod(m_ShapeInferenceMethod);
    layer->SetAllowExpandedDims(m_AllowExpandedDims);

    return layer;
}

} // namespace
