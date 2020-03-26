//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DeviceSpec.hpp"
#include "Graph.hpp"

namespace armnn
{

std::vector<ConvertBf16ToFp32Layer*> InsertConvertBf16ToFp32LayersBefore(Graph& graph,
                                                                         Layer& layer,
                                                                         bool expectCorrectInputType = true);

std::vector<ConvertFp32ToBf16Layer*> InsertConvertFp32ToBf16LayersBefore(Graph& graph,
                                                                         Layer& layer,
                                                                         bool expectCorrectInputType = true);

std::vector<ConvertFp32ToBf16Layer*> InsertConvertFp32ToBf16LayersAfter(Graph& graph, Layer& layer);

std::vector<ConvertFp16ToFp32Layer*> InsertConvertFp16ToFp32LayersBefore(Graph& graph,
                                                                         Layer& layer,
                                                                         bool expectCorrectInputType = true);

std::vector<ConvertFp32ToFp16Layer*> InsertConvertFp32ToFp16LayersAfter(Graph& graph, Layer& layer);

std::vector<DebugLayer*> InsertDebugLayerAfter(Graph& graph, Layer& layer);

} // namespace armnn
