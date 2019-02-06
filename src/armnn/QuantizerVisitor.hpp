//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "LayerVisitorBase.hpp"
#include <armnn/INetwork.hpp>
#include <armnn/Types.hpp>

#include <map>

namespace armnn
{

// Forward declarations
class StaticRangeVisitor;

/// Visitor object for quantizing layers in a network
class QuantizerVisitor : public LayerVisitorBase<VisitorNoThrowPolicy>
{
public:
    QuantizerVisitor(StaticRangeVisitor* ranges);
    ~QuantizerVisitor() = default;

    // Functions to quantize the individual layers, overridden from ILayerVisitor
    void VisitInputLayer(const IConnectableLayer *layer, LayerBindingId id, const char *name = nullptr) override;
    void VisitAdditionLayer(const IConnectableLayer *layer, const char *name = nullptr) override;
    void VisitOutputLayer(const IConnectableLayer *layer, LayerBindingId id, const char *name = nullptr)  override;

    // Extract the quantized network
    INetworkPtr RetrieveFinalNetwork() { return std::move(m_QuantizedNetwork); }
private:

    /// Connects the layer to preceeding layers and sets the quantization parameters based on recorded ranges
    void SetQuantizedInputConnections(const IConnectableLayer *srcLayer, IConnectableLayer *quantizedLayer);

    /// Record the guids so we can easily find the layers later
    void RecordLayer(const IConnectableLayer* srcLayer, IConnectableLayer* qLayer);


    StaticRangeVisitor* m_Ranges;           ///< Previously recorded min/max ranges per intermediate tensor
    INetworkPtr m_QuantizedNetwork;         ///< Quantized version of the model we are building up

    std::map<LayerGuid, LayerGuid> m_OldToNewGuidMap;  ///< Mapping from input network guids to quantized network guids
    std::map<LayerGuid, IConnectableLayer*> m_GuidToLayerMap; ///< Mapping from guid to layer in quantized network
};

} //namespace armnn