//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/LayerVisitorBase.hpp"
#include "RangeTracker.hpp"

#include <armnn/INetwork.hpp>
#include <armnnQuantizer/INetworkQuantizer.hpp>


namespace armnn
{

class StaticRangeStrategy : public IStrategy
{
public:
    StaticRangeStrategy(RangeTracker& rangeTracker);
    ~StaticRangeStrategy() = default;

    void ExecuteStrategy(const armnn::IConnectableLayer *layer,
                         const BaseDescriptor &descriptor,
                         const std::vector<armnn::ConstTensor> &constants,
                         const char *name,
                         const armnn::LayerBindingId id) override;

private:
    /// Set the range for an output slot on a layer
    void SetRange(const IConnectableLayer* layer, unsigned int outputIdx, float min, float max);

    void ForwardParentParameters(const IConnectableLayer* layer);

    /// Mapping from a layer Guid to an array of ranges for outputs
    RangeTracker& m_RangeTracker;

};

} //namespace armnn
