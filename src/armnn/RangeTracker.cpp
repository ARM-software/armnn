//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RangeTracker.hpp"
#include "InternalTypes.hpp"

namespace armnn
{

void RangeTracker::SetRange(const armnn::IConnectableLayer* layer, unsigned int outputIdx, float min, float max)
{
    auto& ranges = m_GuidToRangesMap[layer->GetGuid()];

    unsigned int numOfOutputSlots = layer->GetNumOutputSlots();
    // output layers are a special case
    if (numOfOutputSlots == 0)
    {
        ++numOfOutputSlots;
    }
    if (ranges.size() < numOfOutputSlots)
    {
        ranges.resize(numOfOutputSlots);
    }
    ranges[outputIdx] = std::make_pair(min, max);
}

RangeTracker::MinMaxRange RangeTracker::GetRange(LayerGuid guid, unsigned int idx) const
{
    auto search = m_GuidToRangesMap.find(guid);
    if (search == m_GuidToRangesMap.end())
    {
        if (IsInDynamicMode())
        {
            throw armnn::Exception("Have no entry for layer GUID [" + std::to_string(guid) + "]");
        }
        else
        {
            return DefaultRange();
        }
    }
    return search->second.at(idx);
}

void RangeTracker::RefineMin(LayerGuid guid, unsigned int idx, float newMin)
{
    auto& currentMin = m_GuidToRangesMap.find(guid)->second.at(idx).first;
    if (newMin < currentMin)
    {
        currentMin = newMin;
    }
}

void RangeTracker::RefineMax(LayerGuid guid, unsigned int idx, float newMax)
{
    auto& currentMax = m_GuidToRangesMap.find(guid)->second.at(idx).second;
    if (newMax > currentMax)
    {
        currentMax = newMax;
    }
}

void RangeTracker::ResetMinMax(LayerGuid guid, unsigned int idx, float newMin, float newMax)
{
    auto minMaxPair = m_GuidToRangesMap.find(guid);
    auto& currentMin = minMaxPair->second.at(idx).first;
    auto& currentMax = minMaxPair->second.at(idx).second;

    currentMin = newMin;
    currentMax = newMax;
}

void RangeTracker::Reset()
{
    m_GuidToRangesMap.clear();
}

} //namespace armnn