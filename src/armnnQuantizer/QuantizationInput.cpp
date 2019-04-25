//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "QuantizationInput.hpp"

#include <iostream>
#include <fstream>
#include <cstring>
#include "armnn/Exceptions.hpp"

namespace armnnQuantizer
{

QuantizationInput::QuantizationInput(const unsigned int passId,
    const armnn::LayerBindingId bindingId,
    const std::string fileName):
    m_PassId(passId)
{
    m_LayerBindingIdToFileName.emplace(bindingId, fileName);
}

QuantizationInput::QuantizationInput(const QuantizationInput& other)
{
    m_PassId = other.GetPassId();
    m_LayerBindingIdToFileName.clear();
    for (armnn::LayerBindingId bindingId : other.GetLayerBindingIds())
    {
        std::string filename = other.GetFileName(bindingId);
        AddEntry(bindingId, filename);
    }
}

void QuantizationInput::AddEntry(const armnn::LayerBindingId bindingId, const std::string fileName)
{
    m_LayerBindingIdToFileName.emplace(bindingId, fileName);
}

std::vector<float> QuantizationInput::GetDataForEntry(const armnn::LayerBindingId bindingId) const
{
    if (m_LayerBindingIdToFileName.at(bindingId).empty())
    {
        throw armnn::Exception("Layer binding ID not found");
    }

    std::string fileName = m_LayerBindingIdToFileName.at(bindingId);
    std::ifstream in(fileName.c_str(), std::ifstream::binary);
    if (!in.is_open())
    {
        throw armnn::Exception("Failed to open input tensor file " + fileName);
    }

    std::string line;
    std::vector<float> values;
    char* pEnd;

    while (std::getline(in, line, ' '))
    {
        values.emplace_back(std::strtof(line.c_str(), &pEnd));
    }
    return values;
}

std::vector<armnn::LayerBindingId> QuantizationInput::GetLayerBindingIds() const
{
    std::vector<armnn::LayerBindingId> layerBindingIDs;

    for (auto iterator = m_LayerBindingIdToFileName.begin(); iterator != m_LayerBindingIdToFileName.end(); ++iterator)
    {
        layerBindingIDs.emplace_back(iterator->first);
    }
    return layerBindingIDs;
}

unsigned long QuantizationInput::GetNumberOfInputs() const
{
    return m_LayerBindingIdToFileName.size();
}

unsigned int QuantizationInput::GetPassId() const
{
    return m_PassId;
}

std::string QuantizationInput::GetFileName(const armnn::LayerBindingId bindingId) const
{
    auto iterator = m_LayerBindingIdToFileName.find(bindingId);
    if (iterator != m_LayerBindingIdToFileName.end())
    {
        return m_LayerBindingIdToFileName.at(bindingId);
    }
    else
    {
        throw armnn::Exception("Could not retrieve filename for binding ID " + std::to_string(bindingId));
    }
}

QuantizationInput::~QuantizationInput() noexcept
{
}

}