//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "OverrideInputRangeVisitor.hpp"
#include "NetworkQuantizerUtils.hpp"
#include "Layer.hpp"

#include <armnn/utility/IgnoreUnused.hpp>

namespace armnn
{

OverrideInputRangeVisitor::OverrideInputRangeVisitor(RangeTracker& ranges,
                                                     LayerBindingId layerId,
                                                     const MinMaxRange& minMaxRange)
    : m_Ranges(ranges)
    , m_LayerId(layerId)
    , m_MinMaxRange(minMaxRange)
{}

void OverrideInputRangeVisitor::VisitInputLayer(const IConnectableLayer* layer, LayerBindingId id, const char* name)
{
    IgnoreUnused(name);
    if (m_LayerId == id)
    {
        m_Ranges.SetRange(layer, 0, m_MinMaxRange.first, m_MinMaxRange.second);
    }
}

} // namespace armnn
