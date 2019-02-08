//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RangeTracker.hpp"

namespace armnn
{

void RangeTracker::SetRange(const armnn::IConnectableLayer *layer, unsigned int outputIdx, float min, float max)
{
    auto& ranges = m_GuidToRangesMap[layer->GetGuid()];

    if (ranges.size() < layer->GetNumOutputSlots())
    {
        ranges.resize(layer->GetNumOutputSlots());
    }
    ranges[outputIdx] = std::make_pair(min, max);
}

RangeTracker::MinMaxRange RangeTracker::GetRange(armnn::LayerGuid guid, unsigned int idx) const
{
    auto search = m_GuidToRangesMap.find(guid);
    if (search == m_GuidToRangesMap.end())
    {
        return DefaultRange();
    }
    return search->second.at(idx);
}

} //namespace armnn