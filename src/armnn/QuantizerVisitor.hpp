//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/LayerVisitorBase.hpp"
#include "StaticRangeVisitor.hpp"

#include <armnn/INetwork.hpp>
#include <armnn/Types.hpp>
#include <armnn/INetworkQuantizer.hpp>

#include <unordered_map>

namespace armnn
{

// Forward declaration
class StaticRangeVisitor;

/// Visitor object for quantizing layers in a network
class QuantizerVisitor : public LayerVisitorBase<VisitorNoThrowPolicy>
{
public:
    QuantizerVisitor(const RangeTracker& rangeTracker);
    ~QuantizerVisitor() = default;

    /// Functions to quantize the individual layers, overridden from ILayerVisitor
    void VisitInputLayer(const IConnectableLayer* layer, LayerBindingId id, const char* name = nullptr) override;

    void VisitAdditionLayer(const IConnectableLayer* layer, const char* name = nullptr) override;

    void VisitActivationLayer(const IConnectableLayer* layer,
                              const ActivationDescriptor& activationDescriptor,
                              const char* name = nullptr) override;

    void VisitOutputLayer(const IConnectableLayer* layer, LayerBindingId id, const char* name = nullptr)  override;

    void VisitBatchNormalizationLayer(const IConnectableLayer* layer,
                                      const BatchNormalizationDescriptor& desc,
                                      const ConstTensor& mean,
                                      const ConstTensor& variance,
                                      const ConstTensor& beta,
                                      const ConstTensor& gamma,
                                      const char* name = nullptr) override;

    void VisitFullyConnectedLayer(const IConnectableLayer *layer,
                                  const FullyConnectedDescriptor&,
                                  const ConstTensor&,
                                  const Optional<ConstTensor>&,
                                  const char *name = nullptr)  override;

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

    void VisitSoftmaxLayer(const IConnectableLayer* layer,
                           const SoftmaxDescriptor& softmaxDescriptor,
                           const char* name = nullptr) override;

    /// Extract the quantized network
    INetworkPtr RetrieveFinalNetwork() { return std::move(m_QuantizedNetwork); }

private:
    /// Connects the layer to preceeding layers and sets the quantization parameters based on recorded ranges
    void SetQuantizedInputConnections(const IConnectableLayer* srcLayer, IConnectableLayer* quantizedLayer);

    /// Record the guids so we can easily find the layers later
    void RecordLayer(const IConnectableLayer* srcLayer, IConnectableLayer* qLayer);

    /// Reference to the static range visitor used to retrieve the quantization ranges
    const RangeTracker& m_Ranges;

    /// Quantized version of the model we are building up
    INetworkPtr m_QuantizedNetwork;

    /// Mapping from input network guids to quantized network guids
    std::unordered_map<LayerGuid, LayerGuid> m_OriginalToQuantizedGuidMap;

    /// Mapping from guid to layer in quantized network
    std::unordered_map<LayerGuid, IConnectableLayer*> m_QuantizedGuidToLayerMap;
};

} //namespace armnn
