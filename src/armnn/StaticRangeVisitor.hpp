//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerVisitorBase.hpp"

#include <armnn/INetwork.hpp>
#include <armnn/INetworkQuantizer.hpp>

#include <unordered_map>

namespace armnn
{

/// Visitor class to establish min/max ranges based on the type of the layer
class StaticRangeVisitor : public LayerVisitorBase<VisitorNoThrowPolicy>
{
private:
    using MinMaxRange  = std::pair<float, float>;
    using MinMaxRanges = std::vector<MinMaxRange>;

public:
    StaticRangeVisitor(std::unordered_map<LayerGuid, MinMaxRanges>& guidToRangesMap);
    ~StaticRangeVisitor() = default;

    /// Functions to set the Range on a per-layer-type basis
    void VisitAdditionLayer(const IConnectableLayer* layer, const char* name = nullptr) override;

    void VisitBatchNormalizationLayer(const IConnectableLayer* layer,
                                      const BatchNormalizationDescriptor& desc,
                                      const ConstTensor& mean,
                                      const ConstTensor& variance,
                                      const ConstTensor& beta,
                                      const ConstTensor& gamma,
                                      const char* name = nullptr) override;

    void VisitConvolution2dLayer(const IConnectableLayer* layer,
                                 const Convolution2dDescriptor& convolution2dDescriptor,
                                 const ConstTensor& weights,
                                 const Optional<ConstTensor>& biases,
                                 const char* name = nullptr) override;

    void VisitActivationLayer(const IConnectableLayer* layer,
                              const ActivationDescriptor& activationDescriptor,
                              const char* name = nullptr) override;

    void VisitFullyConnectedLayer(const IConnectableLayer *layer,
                                  const FullyConnectedDescriptor& desc,
                                  const ConstTensor& weights,
                                  const Optional<ConstTensor>& biases,
                                  const char *name) override;

    /// Retrieve the default range
    MinMaxRange DefaultRange() const { return std::make_pair(-15.0f, 15.0f); }

    /// Retrieve the Range for a particular output slot on a particular layer
    MinMaxRange GetRange(LayerGuid guid, unsigned int idx) const;

private:
    /// Set the range for an output slot on a layer
    void SetRange(const IConnectableLayer* layer, unsigned int outputIdx, float min, float max);

    /// Mapping from a layer Guid to an array of ranges for outputs
    std::unordered_map<LayerGuid, MinMaxRanges>& m_GuidToRangesMap;
};

} //namespace armnn
