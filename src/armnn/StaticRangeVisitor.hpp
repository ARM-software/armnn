//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/LayerVisitorBase.hpp"
#include "RangeTracker.hpp"

#include <armnn/INetwork.hpp>
#include <armnn/INetworkQuantizer.hpp>


namespace armnn
{

/// Visitor class to establish min/max ranges based on the type of the layer
class StaticRangeVisitor : public LayerVisitorBase<VisitorNoThrowPolicy>
{
public:
    StaticRangeVisitor(RangeTracker& rangeTracker);
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

    void VisitDepthwiseConvolution2dLayer(const IConnectableLayer* layer,
                                          const DepthwiseConvolution2dDescriptor& desc,
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
    void VisitSoftmaxLayer(const IConnectableLayer* layer,
                           const SoftmaxDescriptor& softmaxDescriptor,
                           const char* name = nullptr) override;

private:
    /// Set the range for an output slot on a layer
    void SetRange(const IConnectableLayer* layer, unsigned int outputIdx, float min, float max);

    /// Mapping from a layer Guid to an array of ranges for outputs
    RangeTracker& m_RangeTracker;
};

} //namespace armnn
