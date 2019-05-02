//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/INetwork.hpp>
#include <armnnQuantizer/INetworkQuantizer.hpp>
#include <armnn/IRuntime.hpp>
#include <armnn/Types.hpp>
#include <armnn/Optional.hpp>

#include "DynamicQuantizationVisitor.hpp"
#include "RangeTracker.hpp"

namespace armnn
{

class NetworkQuantizer : public INetworkQuantizer
{
public:
    NetworkQuantizer(INetwork* inputNetwork, const QuantizerOptions& options)
    : m_InputNetwork(inputNetwork),
      m_NetworkId(0),
      m_Runtime(nullptr, &IRuntime::Destroy),
      m_RefineCount(0),
      m_Options(options) {}

    void OverrideInputRange(LayerBindingId layerId, float min, float max) override;
    void Refine(const InputTensors& inputTensors) override;

    // Required for testing? Need some way to get min/max in RangeTracker (m_Ranges)
    std::pair<float, float> GetMinMaxRange(LayerGuid guid, unsigned int idx) { return m_Ranges.GetRange(guid, idx); }
    INetworkPtr ExportNetwork() override;

private:
    /// Original input network to quantize
    INetwork* m_InputNetwork;

    NetworkId m_NetworkId;

    // if we are run in dynamic mode this unique pointer will hold
    // the runtime between invocations of the Refine method.
    IRuntimePtr m_Runtime;

    Optional<DynamicQuantizationVisitor> m_DynamicQuantizationVisitor;

    // counts the number of times refine is called
    unsigned int m_RefineCount;

    /// Mapping from Guid to an array of ranges for outputs
    RangeTracker m_Ranges;

    /// Options for the NetworkQuantizer
    QuantizerOptions m_Options;

    std::pair<float, float> FindMinMax(ITensorHandle* tensorHandle);
};

} //namespace armnn
