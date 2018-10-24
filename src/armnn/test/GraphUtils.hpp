//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Graph.hpp>

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