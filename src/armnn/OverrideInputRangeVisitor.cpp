//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "OverrideInputRangeVisitor.hpp"
#include "NetworkQuantizerUtils.hpp"
#include "Layer.hpp"

#include <boost/assert.hpp>

namespace armnn
{

OverrideInputRangeVisitor::OverrideInputRangeVisitor(std::unordered_map<LayerGuid, MinMaxRanges>& guidToRangesMap,
                                                     LayerBindingId layerId,
                                                     const MinMaxRange& minMaxRange)
    : m_GuidToRangesMap(guidToRangesMap)
    , m_LayerId(layerId)
    , m_MinMaxRange(minMaxRange)
{}

void OverrideInputRangeVisitor::VisitInputLayer(const IConnectableLayer* layer, LayerBindingId id, const char* name)
{
    if (m_LayerId != id)
    {
        // Not the layer we are looking for
        return;
    }

    SetRange(layer);
}

void OverrideInputRangeVisitor::SetRange(const IConnectableLayer* layer)
{
    BOOST_ASSERT(layer);

    auto& ranges = m_GuidToRangesMap[layer->GetGuid()];

    if (ranges.size() < layer->GetNumOutputSlots())
    {
        ranges.resize(layer->GetNumOutputSlots());
    }
    ranges[0] = m_MinMaxRange;
}

} // namespace armnn
