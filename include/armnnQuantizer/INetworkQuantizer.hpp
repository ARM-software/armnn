//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/INetwork.hpp>
#include <armnn/Types.hpp>
#include <armnn/Tensor.hpp>

namespace armnn
{

struct QuantizerOptions
{
    QuantizerOptions() : QuantizerOptions(DataType::QAsymmU8, false) {}

    QuantizerOptions(DataType activationFormat) : QuantizerOptions(activationFormat, false) {}

    QuantizerOptions(DataType activationFormat, bool preserveType)
    : m_ActivationFormat(activationFormat)
    , m_PreserveType(preserveType) {}

    DataType m_ActivationFormat;
    bool m_PreserveType;
};

using INetworkQuantizerPtr = std::unique_ptr<class INetworkQuantizer, void(*)(INetworkQuantizer* quantizer)>;

/// Quantizer class Quantizes a float32 InputNetwork
class INetworkQuantizer
{
public:
    /// Create Quantizer object and return raw pointer
    static INetworkQuantizer* CreateRaw(INetwork* inputNetwork, const QuantizerOptions& options = QuantizerOptions());

    /// Create Quantizer object wrapped in unique_ptr
    static INetworkQuantizerPtr Create(INetwork* inputNetwork, const QuantizerOptions& options = QuantizerOptions());

    /// Destroy Quantizer object
    static void Destroy(INetworkQuantizer* quantizer);

    /// Overrides the default quantization values for the input layer with the given id
    virtual void OverrideInputRange(LayerBindingId layerId, float min, float max) = 0;

    /// Refine input network with a set of refinement data for specified LayerBindingId
    virtual void Refine(const InputTensors& inputTensors) = 0;

    /// Extract final quantized network
    virtual INetworkPtr ExportNetwork() = 0;

protected:
    virtual ~INetworkQuantizer() {}
};

} //namespace armnn
