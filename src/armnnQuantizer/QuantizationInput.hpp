//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <map>
#include <armnn/Types.hpp>
#include <armnnQuantizer/INetworkQuantizer.hpp>

namespace armnnQuantizer
{

/// QuantizationInput for specific pass ID, can list a corresponding raw data file for each LayerBindingId.
class QuantizationInput
{
public:

    /// Constructor for QuantizationInput
    QuantizationInput(const unsigned int passId,
                      const armnn::LayerBindingId bindingId,
                      const std::string fileName);

    QuantizationInput(const QuantizationInput& other);

    // Add binding ID to image tensor filepath entry
    void AddEntry(const armnn::LayerBindingId bindingId, const std::string fileName);

    // Retrieve tensor data for entry with provided binding ID
    std::vector<float> GetDataForEntry(const armnn::LayerBindingId bindingId) const;

    /// Retrieve Layer Binding IDs for this QuantizationInput.
    std::vector<armnn::LayerBindingId> GetLayerBindingIds() const;

    /// Get number of inputs for this QuantizationInput.
    unsigned long GetNumberOfInputs() const;

    /// Retrieve Pass ID for this QuantizationInput.
    unsigned int GetPassId() const;

    /// Retrieve filename path for specified Layer Binding ID.
    std::string GetFileName(const armnn::LayerBindingId bindingId) const;

    /// Destructor
    ~QuantizationInput() noexcept;

private:
    unsigned int m_PassId;
    std::map<armnn::LayerBindingId, std::string> m_LayerBindingIdToFileName;

};

}