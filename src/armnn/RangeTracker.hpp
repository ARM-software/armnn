//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/INetwork.hpp>
#include <armnn/Types.hpp>

#include <common/include/ProfilingGuid.hpp>

#include <utility>
#include <unordered_map>

namespace armnn
{

class RangeTracker
{
public:
    using MinMaxRange  = std::pair<float, float>;

    /// Retrieve the Range for a particular output slot on a particular layer
    MinMaxRange GetRange(LayerGuid guid, unsigned int idx) const;

    /// Set the range for an output slot on a layer
    void SetRange(const IConnectableLayer* layer, unsigned int outputIdx, float min, float max);

    /// Query function to check that the RangeTracker is empty.
    bool IsEmpty() const { return m_GuidToRangesMap.empty(); }

    /// Query that there is an entry for a layer
    bool HasRanges(LayerGuid guid) const { return m_GuidToRangesMap.find(guid) != m_GuidToRangesMap.end(); }

    /// Update min in RangeTracker with new_min if it is lower than current value
    void RefineMin(LayerGuid guid, unsigned int slotIndex, float newMin);

    /// Update max in RangeTracker with new_max if it is greater than current value
    void RefineMax(LayerGuid guid, unsigned int slotIndex, float newMax);

    /// Overwrite min and max in RangeTracker with newMin and newMax
    void ResetMinMax(LayerGuid guid, unsigned int idx, float newMin, float newMax);

    void Reset();

    void SetDynamicMode(bool flag) { m_DynamicMode = flag; }

    bool IsInDynamicMode() const { return m_DynamicMode; }

private:
    using MinMaxRanges = std::vector<MinMaxRange>;

    /// Retrieve the default range
    MinMaxRange DefaultRange() const { return std::make_pair(-15.0f, 15.0f); }

    /// Mapping from a layer Guid to an array of ranges for outputs
    std::unordered_map<LayerGuid, MinMaxRanges> m_GuidToRangesMap;

    bool m_DynamicMode = false;
};

} //namespace armnn