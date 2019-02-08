//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/INetwork.hpp>
#include <armnn/Types.hpp>

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

private:
    using MinMaxRanges = std::vector<MinMaxRange>;

    /// Retrieve the default range
    MinMaxRange DefaultRange() const { return std::make_pair(-15.0f, 15.0f); }

    /// Mapping from a layer Guid to an array of ranges for outputs
    std::unordered_map<LayerGuid, MinMaxRanges> m_GuidToRangesMap;
};

} //namespace armnn