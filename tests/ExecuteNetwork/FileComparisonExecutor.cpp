//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "FileComparisonExecutor.hpp"
#include <NetworkExecutionUtils/NetworkExecutionUtils.hpp>
#include <algorithm>
#include <filesystem>
#include <iterator>

using namespace armnn;

/**
 * Given a buffer in the expected format. Extract from it the tensor name, tensor type as strings and return an
 * index pointing to the start of the data section.
 *
 * @param buffer data to be parsed.
 * @param tensorName the name of the tensor extracted from the header.
 * @param tensorType the type of the tensor extracted from the header.
 * @return index pointing to the start of the data in the buffer.
 */
unsigned int ExtractHeader(const std::vector<char>& buffer, std::string& tensorName, DataType& tensorType)
{
    auto isColon = [](char c) { return c == ':'; };
    auto isComma = [](char c) { return c == ','; };

    // Find the "," separator marks the end of the tensor name.
    auto firstComma = std::find_if(buffer.begin(), buffer.end(), isComma);
    if (firstComma == buffer.end())
    {
        throw ParseException("Unable to read tensor name from file.");
    }
    tensorName.assign(buffer.begin(), firstComma);

    // The next colon marks the end of the data type string.
    auto endOfHeader = std::find_if(firstComma, buffer.end(), isColon);
    if (firstComma == buffer.end())
    {
        throw ParseException("Unable to read tensor type from file.");
    }
    std::string type(++firstComma, endOfHeader);
    // Remove any leading or trailing whitespace.
    type.erase(remove_if(type.begin(), type.end(), isspace), type.end());
    if (type == "Float16")
    {
        tensorType = DataType::Float16;
    }
    else if (type == "Float32")
    {
        tensorType = DataType::Float32;
    }
    else if (type == "QAsymmU8")
    {
        tensorType = DataType::QAsymmU8;
    }
    else if (type == "Signed32")
    {
        tensorType = DataType::Signed32;
    }
    else if (type == "Boolean")
    {
        tensorType = DataType::Boolean;
    }
    else if (type == "QSymmS16")
    {
        tensorType = DataType::QSymmS16;
    }
    else if (type == "QSymmS8")
    {
        tensorType = DataType::QSymmS8;
    }
    else if (type == "QAsymmS8")
    {
        tensorType = DataType::QAsymmS8;
    }
    else if (type == "BFloat16")
    {
        tensorType = DataType::BFloat16;
    }
    else if (type == "Signed64")
    {
        tensorType = DataType::Signed64;
    }
    else
    {
        throw ParseException("Invalid data type in header.");
    }
    // Remember to move the iterator past the colon.
    return (++endOfHeader - buffer.begin());
}

/**
 * Extract the data from the file and return as a typed vector of elements.
 *
 * @param buffer data to be parsed.
 * @param dataStart Index into the vector where the tensor data starts.
 * @param tensorType the type of the tensor extracted from the header.
 */
template <typename T>
void ReadData(const std::vector<char>& buffer,
              const unsigned int dataStart,
              const DataType& tensorType,
              std::vector<T>& results)
{
    unsigned int index = dataStart;
    while (index < buffer.size())
    {
        std::string elementString;
        // Extract into a string until the next space.
        while (index < buffer.size() && buffer[index] != ' ')
        {
            elementString.push_back(buffer[index]);
            index++;
        }
        if (!elementString.empty())
        {
            switch (tensorType)
            {
                case DataType::Float32: {
                    results.push_back(std::stof(elementString));
                    break;
                }

                case DataType::Signed32: {
                    results.push_back(std::stoi(elementString));
                    break;
                }
                case DataType::QSymmS8:
                case DataType::QAsymmS8: {
                    results.push_back(elementString[0]);
                    break;
                }
                case DataType::QAsymmU8: {
                    results.push_back(elementString[0]);
                    break;
                }
                case DataType::Float16:
                case DataType::QSymmS16:
                case DataType::BFloat16:
                case DataType::Boolean:
                case DataType::Signed64:
                default: {
                    LogAndThrow("Unsupported DataType");
                }
            }
            // Finally, skip the space we know is there.
            index++;
        }
        else
        {
            if (index < buffer.size())
            {
                index++;
            }
        }
    }
}

/**
 * Open the given file and read the data out of it to construct a Tensor. This could throw FileNotFoundException
 * or InvalidArgumentException
 *
 * @param fileName the file to be read.
 * @return a populated tensor.
 */
Tensor ReadTensorFromFile(const std::string fileName)
{
    if (!std::filesystem::exists(fileName))
    {
        throw FileNotFoundException("The file \"" + fileName + "\" could not be found.");
    }
    // The format we are reading in is based on NetworkExecutionUtils::WriteToFile. This could potentially
    // be an enormous tensor. We'll limit what we can read in to 1Mb.
    std::uintmax_t maxFileSize = 1048576;
    std::uintmax_t fileSize    = std::filesystem::file_size(fileName);
    if (fileSize > maxFileSize)
    {
        throw InvalidArgumentException("The file \"" + fileName + "\" exceeds max size of 1 Mb.");
    }

    // We'll read the entire file into one buffer.
    std::ifstream file(fileName, std::ios::binary);
    std::vector<char> buffer(fileSize);
    if (file.read(buffer.data(), fileSize))
    {
        std::string tensorName;
        DataType tensorType;
        unsigned int tensorDataStart = ExtractHeader(buffer, tensorName, tensorType);
        switch (tensorType)
        {
            case DataType::Float32: {
                std::vector<float> floatVector;
                ReadData(buffer, tensorDataStart, tensorType, floatVector);
                TensorInfo info({ static_cast<unsigned int>(floatVector.size()), 1, 1, 1 }, DataType::Float32);
                float* floats = new float[floatVector.size()];
                memcpy(floats, floatVector.data(), (floatVector.size() * sizeof(float)));
                return Tensor(info, floats);
            }
            case DataType::Signed32: {
                std::vector<int> intVector;
                ReadData(buffer, tensorDataStart, tensorType, intVector);
                TensorInfo info({ static_cast<unsigned int>(intVector.size()), 1, 1, 1 }, DataType::Signed32);
                int* ints = new int[intVector.size()];
                memcpy(ints, intVector.data(), (intVector.size() * sizeof(float)));
                return Tensor(info, ints);
            }
            case DataType::QSymmS8: {
                std::vector<int8_t> intVector;
                ReadData(buffer, tensorDataStart, tensorType, intVector);
                TensorInfo info({ static_cast<unsigned int>(intVector.size()), 1, 1, 1 }, DataType::QSymmS8);
                int8_t* ints = new int8_t[intVector.size()];
                memcpy(ints, intVector.data(), (intVector.size() * sizeof(float)));
                return Tensor(info, ints);
            }
            case DataType::QAsymmS8: {
                std::vector<int8_t> intVector;
                ReadData(buffer, tensorDataStart, tensorType, intVector);
                TensorInfo info({ static_cast<unsigned int>(intVector.size()), 1, 1, 1 }, DataType::QAsymmS8);
                int8_t* ints = new int8_t[intVector.size()];
                memcpy(ints, intVector.data(), (intVector.size() * sizeof(float)));
                return Tensor(info, ints);
            }
            case DataType::QAsymmU8: {
                std::vector<uint8_t> intVector;
                ReadData(buffer, tensorDataStart, tensorType, intVector);
                TensorInfo info({ static_cast<unsigned int>(intVector.size()), 1, 1, 1 }, DataType::QAsymmU8);
                uint8_t* ints = new uint8_t[intVector.size()];
                memcpy(ints, intVector.data(), (intVector.size() * sizeof(float)));
                return Tensor(info, ints);
            }
            default:
                throw InvalidArgumentException("The tensor data could not be read from \"" + fileName + "\"");
        }
    }
    else
    {
        throw ParseException("Filed to read the contents of \"" + fileName + "\"");
    }

    Tensor result;
    return result;
}

FileComparisonExecutor::FileComparisonExecutor(const ExecuteNetworkParams& params)
    : m_Params(params)
{}

std::vector<const void*> FileComparisonExecutor::Execute()
{
    std::string filesToCompare = this->m_Params.m_ComparisonFile;
    if (filesToCompare.empty())
    {
        throw InvalidArgumentException("The file(s) to compare was not set.");
    }
    // filesToCompare is one or more files containing output tensors. Iterate and read in the tensors.
    // We'll assume the string follows the same comma seperated format as write-outputs-to-file.
    std::stringstream ss(filesToCompare);
    std::vector<std::string> fileNames;
    std::string errorString;
    while (ss.good())
    {
        std::string substr;
        getline(ss, substr, ',');
        // Check the file exist.
        if (!std::filesystem::exists(substr))
        {
            errorString += substr + " ";
        }
        else
        {
            fileNames.push_back(substr);
        }
    }
    if (!errorString.empty())
    {
        throw FileNotFoundException("The following file(s) to compare could not be found: " + errorString);
    }
    // Read in the tensors into m_OutputTensorsVec
    OutputTensors outputs;
    std::vector<const void*> results;
    for (auto file : fileNames)
    {
        Tensor t = ReadTensorFromFile(file);
        outputs.push_back({ 0, Tensor(t.GetInfo(), t.GetMemoryArea()) });
        results.push_back(t.GetMemoryArea());
    }
    m_OutputTensorsVec.push_back(outputs);
    return results;
}

void FileComparisonExecutor::PrintNetworkInfo()
{
    std::cout << "Not implemented in this class." << std::endl;
}

void FileComparisonExecutor::CompareAndPrintResult(std::vector<const void*> otherOutput)
{
    unsigned int index = 0;
    std::string typeString;
    for (const auto& outputTensors : m_OutputTensorsVec)
    {
        for (const auto& outputTensor : outputTensors)
        {
            size_t size   = outputTensor.second.GetNumBytes();
            double result = ComputeByteLevelRMSE(outputTensor.second.GetMemoryArea(), otherOutput[index++], size);
            std::cout << "Byte level root mean square error: " << result << "\n";
        }
    }
}

FileComparisonExecutor::~FileComparisonExecutor()
{
    // If there are tensors defined in m_OutputTensorsVec we need to clean up their memory usage.
    for (OutputTensors opTensor : m_OutputTensorsVec)
    {
        for (std::pair<LayerBindingId, class Tensor> pair : opTensor)
        {
            Tensor t = pair.second;
            // Based on the tensor type and size recover the memory.
            switch (t.GetDataType())
            {
                case DataType::Float32:
                    delete[] static_cast<float*>(t.GetMemoryArea());
                    break;
                case DataType::Signed32:
                    delete[] static_cast<int*>(t.GetMemoryArea());
                    break;
                case DataType::QSymmS8:
                    delete[] static_cast<int8_t*>(t.GetMemoryArea());
                    break;
                case DataType::QAsymmS8:
                    delete[] static_cast<int8_t*>(t.GetMemoryArea());
                    break;
                case DataType::QAsymmU8:
                    delete[] static_cast<uint8_t*>(t.GetMemoryArea());
                    break;
                default:
                    std::cout << "The data type wasn't created in ReadTensorFromFile" << std::endl;
            }
        }
    }

}
