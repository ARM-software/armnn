//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DeviceSpec.hpp"
#include "Graph.hpp"

namespace armnn
{

std::vector<ConvertFp16ToFp32Layer*> InsertConvertFp16ToFp32LayersBefore(Graph& graph,
                                                                         Layer& layer,
                                                                         bool expectCorrectInputType = true);

std::vector<ConvertFp32ToFp16Layer*> InsertConvertFp32ToFp16LayersAfter(Graph& graph, Layer& layer);

std::vector<DebugLayer*> InsertDebugLayerAfter(Graph& graph, Layer& layer, bool toFile);

bool RevertConstantWeightsToFP32(Layer* layer);

} // namespace armnn
