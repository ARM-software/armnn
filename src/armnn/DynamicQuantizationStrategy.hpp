//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "armnn/LayerVisitorBase.hpp"
#include "RangeTracker.hpp"
#include "layers/DebugLayer.hpp"

#include <armnn/INetwork.hpp>
#include <armnnQuantizer/INetworkQuantizer.hpp>

namespace armnn
{

/// Visitor class implementation to gather the TensorInfo for LayerBindingID for creation of ConstTensor for Refine.
class DynamicQuantizationStrategy : public armnn::IStrategy
{
public:

    DynamicQuantizationStrategy(RangeTracker& rangeTracker, Graph& graph);
    ~DynamicQuantizationStrategy() = default;

    virtual void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                                 const armnn::BaseDescriptor& descriptor,
                                 const std::vector<armnn::ConstTensor>& constants,
                                 const char* name,
                                 const armnn::LayerBindingId id = 0) override;

    const std::vector<armnn::LayerBindingId>& GetOutputLayers();
    void VisitNonCalibratedLayers();
    void FinishStrategy() override;


private:
    /// Set the range for an output slot on a layer
    void SetRange(const IConnectableLayer* layer, unsigned int outputIdx, float min, float max);

    void ForwardParentParameters(const IConnectableLayer* layer);

    /// Mapping from a layer Guid to an array of ranges for outputs
    RangeTracker& m_RangeTracker;

    Graph& m_Graph;

    std::vector<const IConnectableLayer*> m_LayersToCalibrate;
    std::vector<const IConnectableLayer*> m_LayersNotToCalibrate;
    std::vector<DebugLayer*> m_DebugLayers;

    std::vector<armnn::LayerBindingId> m_OutputLayers;
    void AddToCalibratedLayers(const IConnectableLayer* layer);
    void AddToNonCalibratedLayers(const IConnectableLayer* layer);
    void RemoveDebugLayers();


};
} //namespace armnn
