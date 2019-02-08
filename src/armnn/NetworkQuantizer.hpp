//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/INetwork.hpp>
#include <armnn/INetworkQuantizer.hpp>
#include <armnn/Types.hpp>

#include "RangeTracker.hpp"

namespace armnn
{

class NetworkQuantizer : public INetworkQuantizer
{
public:
    NetworkQuantizer(INetwork* inputNetwork) : m_InputNetwork(inputNetwork) {}

    void OverrideInputRange(LayerBindingId layerId, float min, float max) override;
    INetworkPtr ExportNetwork() override;

private:
    /// Original input network to quantize
    INetwork* m_InputNetwork;

    /// Mapping from Guid to an array of ranges for outputs
    RangeTracker m_Ranges;
};

} //namespace armnn
