//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/INetwork.hpp>

namespace armnn
{

using INetworkQuantizerPtr = std::unique_ptr<class INetworkQuantizer, void(*)(INetworkQuantizer* quantizer)>;

/// Quantizer class Quantizes a float32 InputNetwork
class INetworkQuantizer
{
public:
    static INetworkQuantizer* CreateRaw(INetwork* inputNetwork); ///< Create Quantizer object and return raw pointer
    static INetworkQuantizerPtr Create(INetwork* inputNetwork);  ///< Create Quantizer object wrapped in unique_ptr
    static void Destroy(INetworkQuantizer* quantizer);           ///< Destroy Quantizer object

    /// Overrides the default quantization values for the input layer with the given id
    virtual void OverrideInputRange(LayerBindingId layerId, float min, float max) = 0;

    /// Extract final quantized network
    virtual INetworkPtr ExportNetwork() = 0;

protected:
    virtual ~INetworkQuantizer() {}
};

} //namespace armnn
