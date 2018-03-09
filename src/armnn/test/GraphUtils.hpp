//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "Graph.hpp"
#include <string>

namespace
{

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
}