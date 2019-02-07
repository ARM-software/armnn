//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/INetwork.hpp>
#include <armnn/INetworkQuantizer.hpp>
#include <armnn/Types.hpp>

#include <unordered_map>

namespace armnn
{

class NetworkQuantizer : public INetworkQuantizer
{
public:
    NetworkQuantizer(INetwork* inputNetwork) : m_InputNetwork(inputNetwork) {}

    void OverrideInputRange(LayerBindingId layerId, float min, float max) override;
    INetworkPtr ExportNetwork() override;

private:
    using MinMaxRange  = std::pair<float, float>;
    using MinMaxRanges = std::vector<MinMaxRange>;

    INetwork* m_InputNetwork;

    /// Mapping from Guid to an array of ranges for outputs
    std::unordered_map<LayerGuid, MinMaxRanges> m_GuidToRangesMap;
};

} //namespace armnn
