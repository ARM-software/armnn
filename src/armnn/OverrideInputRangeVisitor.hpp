//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "NetworkQuantizer.hpp"
#include "armnn/LayerVisitorBase.hpp"
#include "RangeTracker.hpp"

#include <unordered_map>

namespace armnn
{
class OverrideInputRangeStrategy : public IStrategy
{
private:
    using MinMaxRange  = RangeTracker::MinMaxRange;
public :
    OverrideInputRangeStrategy(RangeTracker& ranges,
                               LayerBindingId layerId,
                               const MinMaxRange& minMaxRange)
                               : m_Ranges(ranges)
            , m_LayerId(layerId)
            , m_MinMaxRange(minMaxRange){}

    ~OverrideInputRangeStrategy() = default;

    void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                         const BaseDescriptor& descriptor,
                         const std::vector<armnn::ConstTensor>& constants,
                         const char* name,
                         const armnn::LayerBindingId id) override
    {
        IgnoreUnused(name, constants, id, descriptor);

        switch (layer->GetType())
        {
            case armnn::LayerType::Input :
            {
                if (m_LayerId == id)
                {
                    m_Ranges.SetRange(layer, 0, m_MinMaxRange.first, m_MinMaxRange.second);
                }
                break;
            }
            default:
            {
                std::cout << "dont know this one" << std::endl;
            }
        }
    }

private:
    /// Mapping from a layer Guid to an array of ranges for outputs
    RangeTracker& m_Ranges;

    /// The id of the input layer of which to override the input range
    LayerBindingId m_LayerId;

    /// The new input range to be applied to the input layer
    MinMaxRange m_MinMaxRange;
};



/// Visitor object for overriding the input range of the quantized input layers in a network
class OverrideInputRangeVisitor : public LayerVisitorBase<VisitorNoThrowPolicy>
{
private:
    using MinMaxRange  = RangeTracker::MinMaxRange;

public:
    OverrideInputRangeVisitor(RangeTracker& ranges,
                              LayerBindingId layerId,
                              const MinMaxRange& minMaxRange);
    ~OverrideInputRangeVisitor() = default;

    void VisitInputLayer(const IConnectableLayer* layer, LayerBindingId id, const char* name = nullptr) override;

private:
    /// Mapping from a layer Guid to an array of ranges for outputs
    RangeTracker& m_Ranges;

    /// The id of the input layer of which to override the input range
    LayerBindingId m_LayerId;

    /// The new input range to be applied to the input layer
    MinMaxRange m_MinMaxRange;
};

} // namespace armnn
