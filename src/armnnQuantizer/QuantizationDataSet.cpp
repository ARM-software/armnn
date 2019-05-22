//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "QuantizationDataSet.hpp"
#include "CsvReader.hpp"

#define BOOST_FILESYSTEM_NO_DEPRECATED

#include <boost/filesystem/operations.hpp>
#include <boost/filesystem/path.hpp>

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


unsigned int GetPassIdFromCsvRow(std::vector<armnnUtils::CsvRow> csvRows, unsigned int rowIndex)
{
    unsigned int passId;
    try
    {
        passId = static_cast<unsigned int>(std::stoi(csvRows[rowIndex].values[0]));
    }
    catch (const std::invalid_argument&)
    {
        throw armnn::ParseException("Pass ID [" + csvRows[rowIndex].values[0] + "]" +
                                    " is not correct format on CSV row " + std::to_string(rowIndex));
    }
    return passId;
}

armnn::LayerBindingId GetBindingIdFromCsvRow(std::vector<armnnUtils::CsvRow> csvRows, unsigned int rowIndex)
{
    armnn::LayerBindingId bindingId;
    try
    {
        bindingId = std::stoi(csvRows[rowIndex].values[1]);
    }
    catch (const std::invalid_argument&)
    {
        throw armnn::ParseException("Binding ID [" + csvRows[rowIndex].values[0] + "]" +
                                    " is not correct format on CSV row " + std::to_string(rowIndex));
    }
    return bindingId;
}

std::string GetFileNameFromCsvRow(std::vector<armnnUtils::CsvRow> csvRows, unsigned int rowIndex)
{
    std::string fileName = csvRows[rowIndex].values[2];

    if (!boost::filesystem::exists(fileName))
    {
        throw armnn::ParseException("File [ " + fileName + "] provided on CSV row " + std::to_string(rowIndex) +
                                    " does not exist.");
    }

    if (fileName.empty())
    {
        throw armnn::ParseException("Filename cannot be empty on CSV row " + std::to_string(rowIndex));
    }
    return fileName;
}


void QuantizationDataSet::ParseCsvFile()
{
    std::map<unsigned int, QuantizationInput> passIdToQuantizationInput;
    armnnUtils::CsvReader reader;

    if (m_CsvFilePath == "")
    {
        throw armnn::Exception("CSV file not specified.");
    }

    // Parse CSV file and extract data
    std::vector<armnnUtils::CsvRow> csvRows = reader.ParseFile(m_CsvFilePath);
    if (csvRows.empty())
    {
        throw armnn::Exception("CSV file [" + m_CsvFilePath + "] is empty.");
    }

    for (unsigned int i = 0; i < csvRows.size(); ++i)
    {
        if (csvRows[i].values.size() != 3)
        {
            throw armnn::Exception("CSV file [" + m_CsvFilePath + "] does not have correct number of entries " +
                                   "on line " + std::to_string(i) + ". Expected 3 entries " +
                                   "but was " + std::to_string(csvRows[i].values.size()));
        }

        unsigned int passId = GetPassIdFromCsvRow(csvRows, i);
        armnn::LayerBindingId bindingId = GetBindingIdFromCsvRow(csvRows, i);
        std::string rawFileName = GetFileNameFromCsvRow(csvRows, i);

        AddInputData(passId, bindingId, rawFileName, passIdToQuantizationInput);
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
