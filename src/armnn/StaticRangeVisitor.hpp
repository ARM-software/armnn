//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerVisitorBase.hpp"

#include <armnn/INetwork.hpp>

#include <map>
#include <vector>

namespace armnn
{

/// Visitor class to establish min/max ranges based on the type of the layer
class StaticRangeVisitor : public LayerVisitorBase<VisitorNoThrowPolicy>
{
public:
    StaticRangeVisitor() = default;
    ~StaticRangeVisitor() = default;

    using MinMaxRange = std::pair<float, float>;
    using MinMaxRanges = std::vector<MinMaxRange>;

    /// Functions to set the Range on a per-layer-type basis
    void VisitAdditionLayer(const IConnectableLayer *layer, const char *name = nullptr) override;
    void VisitBatchNormalizationLayer(const IConnectableLayer* layer,
                                      const BatchNormalizationDescriptor& desc,
                                      const ConstTensor& mean,
                                      const ConstTensor& variance,
                                      const ConstTensor& beta,
                                      const ConstTensor& gamma,
                                      const char* name = nullptr) override;
    void VisitActivationLayer(const IConnectableLayer *layer,
                              const ActivationDescriptor& activationDescriptor,
                              const char *name = nullptr) override;

    /// Retreive the default range
    MinMaxRange DefaultRange() const { return std::make_pair(-15.0f, 15.0f); }

    /// Retreive the Range for a particular output slot on a particular layer
    MinMaxRange GetRange(LayerGuid guid, unsigned int idx) const;

private:
    /// Set the range for an output slot on a layer
    void SetRange(const IConnectableLayer* layer, unsigned int outputIdx, float min, float max);

    /// Mapping from Guid to an array of ranges for outputs
    std::map<LayerGuid, MinMaxRanges> m_GuidToRangesMap;
};

} //namespace armnn