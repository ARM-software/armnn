//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NetworkQuantizer.hpp"
#include "LayerVisitorBase.hpp"

#include <unordered_map>

namespace armnn
{

/// Visitor object for overriding the input range of the quantized input layers in a network
class OverrideInputRangeVisitor : public LayerVisitorBase<VisitorNoThrowPolicy>
{
private:
    using MinMaxRange  = std::pair<float, float>;
    using MinMaxRanges = std::vector<MinMaxRange>;

public:
    OverrideInputRangeVisitor(std::unordered_map<LayerGuid, MinMaxRanges>& guidToRangesMap,
                              LayerBindingId layerId,
                              const MinMaxRange& minMaxRange);
    ~OverrideInputRangeVisitor() = default;

    void VisitInputLayer(const IConnectableLayer *layer, LayerBindingId id, const char *name = nullptr) override;

private:
    /// Sets the range for the given input layer
    void SetRange(const IConnectableLayer* layer);

    /// Mapping from a layer Guid to an array of ranges for outputs
    std::unordered_map<LayerGuid, MinMaxRanges>& m_GuidToRangesMap;

    /// The id of the input layer of which to override the input range
    LayerBindingId m_LayerId;

    /// The new input range to be applied to the input layer
    MinMaxRange m_MinMaxRange;
};

} // namespace armnn
