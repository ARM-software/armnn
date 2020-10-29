//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "QuantizationDataSet.hpp"

#include <fmt/format.h>

#include <armnn/utility/StringUtils.hpp>
#include <armnn/utility/IgnoreUnused.hpp>
#include <Filesystem.hpp>

namespace armnnQuantizer
{

QuantizationDataSet::QuantizationDataSet()
{
}

QuantizationDataSet::QuantizationDataSet(const std::string csvFilePath):
    m_QuantizationInputs(),
    m_CsvFilePath(csvFilePath)
{
    ParseCsvFile();
}

void AddInputData(unsigned int passId,
                  armnn::LayerBindingId bindingId,
                  const std::string& inputFilePath,
                  std::map<unsigned int, QuantizationInput>& passIdToQuantizationInput)
{
    auto iterator = passIdToQuantizationInput.find(passId);
    if (iterator == passIdToQuantizationInput.end())
    {
        QuantizationInput input(passId, bindingId, inputFilePath);
        passIdToQuantizationInput.emplace(passId, input);
    }
    else
    {
        auto existingQuantizationInput = iterator->second;
        existingQuantizationInput.AddEntry(bindingId, inputFilePath);
    }
}

QuantizationDataSet::~QuantizationDataSet()
{
}

void InputLayerVisitor::VisitInputLayer(const armnn::IConnectableLayer* layer,
                                        armnn::LayerBindingId id,
                                        const char* name)
{
    armnn::IgnoreUnused(name);
    m_TensorInfos.emplace(id, layer->GetOutputSlot(0).GetTensorInfo());
}

armnn::TensorInfo InputLayerVisitor::GetTensorInfo(armnn::LayerBindingId layerBindingId)
{
    auto iterator = m_TensorInfos.find(layerBindingId);
    if (iterator != m_TensorInfos.end())
    {
        return m_TensorInfos.at(layerBindingId);
    }
    else
    {
        throw armnn::Exception("Could not retrieve tensor info for binding ID " + std::to_string(layerBindingId));
    }
}


unsigned int GetPassIdFromCsvRow(std::vector<std::string> tokens, unsigned int lineIndex)
{
    unsigned int passId;
    try
    {
        passId = static_cast<unsigned int>(std::stoi(tokens[0]));
    }
    catch (const std::invalid_argument&)
    {
        throw armnn::ParseException(fmt::format("Pass ID [{}] is not correct format on CSV row {}",
                                                tokens[0], lineIndex));
    }
    return passId;
}

armnn::LayerBindingId GetBindingIdFromCsvRow(std::vector<std::string> tokens, unsigned int lineIndex)
{
    armnn::LayerBindingId bindingId;
    try
    {
        bindingId = std::stoi(tokens[1]);
    }
    catch (const std::invalid_argument&)
    {
        throw armnn::ParseException(fmt::format("Binding ID [{}] is not correct format on CSV row {}",
                                                tokens[1], lineIndex));
    }
    return bindingId;
}

std::string GetFileNameFromCsvRow(std::vector<std::string> tokens, unsigned int lineIndex)
{
    std::string fileName = armnn::stringUtils::StringTrim(tokens[2]);

    if (!fs::exists(fileName))
    {
        throw armnn::ParseException(fmt::format("File [{}] provided on CSV row {} does not exist.",
                                                fileName, lineIndex));
    }

    if (fileName.empty())
    {
        throw armnn::ParseException(fmt::format("Filename cannot be empty on CSV row {} ", lineIndex));
    }
    return fileName;
}


void QuantizationDataSet::ParseCsvFile()
{
    std::map<unsigned int, QuantizationInput> passIdToQuantizationInput;

    if (m_CsvFilePath == "")
    {
        throw armnn::Exception("CSV file not specified.");
    }

    std::ifstream inf (m_CsvFilePath.c_str());
    std::string line;
    std::vector<std::string> tokens;
    unsigned int lineIndex = 0;

    if (!inf)
    {
        throw armnn::Exception(fmt::format("CSV file {} not found.", m_CsvFilePath));
    }

    while (getline(inf, line))
    {
        tokens = armnn::stringUtils::StringTokenizer(line, ",");

        if (tokens.size() != 3)
        {
            throw armnn::Exception(fmt::format("CSV file [{}] does not have correct number of entries" \
                                               "on line {}. Expected 3 entries but was {}.",
                                               m_CsvFilePath, lineIndex, tokens.size()));

        }

        unsigned int passId = GetPassIdFromCsvRow(tokens, lineIndex);
        armnn::LayerBindingId bindingId = GetBindingIdFromCsvRow(tokens, lineIndex);
        std::string rawFileName = GetFileNameFromCsvRow(tokens, lineIndex);

        AddInputData(passId, bindingId, rawFileName, passIdToQuantizationInput);

        ++lineIndex;
    }

    if (passIdToQuantizationInput.empty())
    {
        throw armnn::Exception("Could not parse CSV file.");
    }

    // Once all entries in CSV file are parsed successfully and QuantizationInput map is populated, populate
    // QuantizationInputs iterator for easier access and clear the map
    for (auto itr = passIdToQuantizationInput.begin(); itr != passIdToQuantizationInput.end(); ++itr)
    {
        m_QuantizationInputs.emplace_back(itr->second);
    }
}

}
