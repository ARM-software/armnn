//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NetworkExecutionUtils.hpp"

#include <armnnUtils/Filesystem.hpp>
#include <InferenceTest.hpp>
#include <ResolveType.hpp>

#if defined(ARMNN_SERIALIZER)
#include "armnnDeserializer/IDeserializer.hpp"
#endif
#if defined(ARMNN_TF_LITE_PARSER)
#include "armnnTfLiteParser/ITfLiteParser.hpp"
#endif
#if defined(ARMNN_ONNX_PARSER)
#include "armnnOnnxParser/IOnnxParser.hpp"
#endif

template<armnn::DataType NonQuantizedType>
auto ParseDataArray(std::istream& stream);

template<armnn::DataType QuantizedType>
auto ParseDataArray(std::istream& stream,
                    const float& quantizationScale,
                    const int32_t& quantizationOffset);

template<>
auto ParseDataArray<armnn::DataType::Float32>(std::istream& stream)
{
    return ParseArrayImpl<float>(stream, [](const std::string& s) { return std::stof(s); });
}

template<>
auto ParseDataArray<armnn::DataType::Signed32>(std::istream& stream)
{
    return ParseArrayImpl<int>(stream, [](const std::string& s) { return std::stoi(s); });
}

template<>
auto ParseDataArray<armnn::DataType::QAsymmS8>(std::istream& stream)
{
    return ParseArrayImpl<int8_t>(stream,
                                  [](const std::string& s) { return armnn::numeric_cast<int8_t>(std::stoi(s)); });
}

template<>
auto ParseDataArray<armnn::DataType::QAsymmU8>(std::istream& stream)
{
    return ParseArrayImpl<uint8_t>(stream,
                                   [](const std::string& s) { return armnn::numeric_cast<uint8_t>(std::stoi(s)); });
}


template<>
auto ParseDataArray<armnn::DataType::QSymmS8>(std::istream& stream)
{
    return ParseArrayImpl<int8_t>(stream,
                                   [](const std::string& s) { return armnn::numeric_cast<int8_t>(std::stoi(s)); });
}

template<>
auto ParseDataArray<armnn::DataType::QAsymmS8>(std::istream& stream,
                                               const float& quantizationScale,
                                               const int32_t& quantizationOffset)
{
    return ParseArrayImpl<int8_t>(stream,
                                  [&quantizationScale, &quantizationOffset](const std::string& s)
                                  {
                                      return armnn::numeric_cast<int8_t>(
                                              armnn::Quantize<int8_t>(std::stof(s),
                                                                      quantizationScale,
                                                                      quantizationOffset));
                                  });
}

template<>
auto ParseDataArray<armnn::DataType::QAsymmU8>(std::istream& stream,
                                               const float& quantizationScale,
                                               const int32_t& quantizationOffset)
{
    return ParseArrayImpl<uint8_t>(stream,
                                   [&quantizationScale, &quantizationOffset](const std::string& s)
                                   {
                                       return armnn::numeric_cast<uint8_t>(
                                               armnn::Quantize<uint8_t>(std::stof(s),
                                                                        quantizationScale,
                                                                        quantizationOffset));
                                   });
}

template<armnn::DataType ArmnnType, typename T = armnn::ResolveType<ArmnnType>>
std::vector<T> GenerateDummyTensorData(unsigned int numElements)
{
    return std::vector<T>(numElements, static_cast<T>(0));
}


std::vector<unsigned int> ParseArray(std::istream& stream)
{
    return ParseArrayImpl<unsigned int>(
            stream,
            [](const std::string& s) { return armnn::numeric_cast<unsigned int>(std::stoi(s)); });
}

std::vector<std::string> ParseStringList(const std::string& inputString, const char* delimiter)
{
    std::stringstream stream(inputString);
    return ParseArrayImpl<std::string>(stream, [](const std::string& s) {
        return armnn::stringUtils::StringTrimCopy(s); }, delimiter);
}


TensorPrinter::TensorPrinter(const std::string& binding,
                             const armnn::TensorInfo& info,
                             const std::string& outputTensorFile,
                             bool dequantizeOutput,
                             const bool printToConsole)
                             : m_OutputBinding(binding)
                             , m_Scale(info.GetQuantizationScale())
                             , m_Offset(info.GetQuantizationOffset())
                             , m_OutputTensorFile(outputTensorFile)
                             , m_DequantizeOutput(dequantizeOutput)
                             , m_PrintToConsole(printToConsole) {}

void TensorPrinter::operator()(const std::vector<float>& values)
{
    if (m_PrintToConsole)
    {
        std::cout << m_OutputBinding << ": ";
        ForEachValue(values, [](float value)
        {
            printf("%f ", value);
        });
        printf("\n");
    }
    WriteToFile(values);
}

void TensorPrinter::operator()(const std::vector<uint8_t>& values)
{
    if(m_DequantizeOutput)
    {
        auto& scale = m_Scale;
        auto& offset = m_Offset;
        std::vector<float> dequantizedValues;
        ForEachValue(values, [&scale, &offset, &dequantizedValues](uint8_t value)
        {
            auto dequantizedValue = armnn::Dequantize(value, scale, offset);
            dequantizedValues.push_back(dequantizedValue);
        });

        if (m_PrintToConsole)
        {
            std::cout << m_OutputBinding << ": ";
            ForEachValue(dequantizedValues, [](float value)
            {
                printf("%f ", value);
            });
            printf("\n");
        }

        WriteToFile(dequantizedValues);
    }
    else
    {
        const std::vector<int> intValues(values.begin(), values.end());
        operator()(intValues);
    }
}

void TensorPrinter::operator()(const std::vector<int8_t>& values)
{
    if (m_PrintToConsole)
    {
        std::cout << m_OutputBinding << ": ";
        ForEachValue(values, [](int8_t value)
        {
            printf("%d ", value);
        });
        printf("\n");
    }
    WriteToFile(values);
}

void TensorPrinter::operator()(const std::vector<int>& values)
{
    if (m_PrintToConsole)
    {
        std::cout << m_OutputBinding << ": ";
        ForEachValue(values, [](int value)
        {
            printf("%d ", value);
        });
        printf("\n");
    }
    WriteToFile(values);
}

template<typename Container, typename Delegate>
void TensorPrinter::ForEachValue(const Container& c, Delegate delegate)
{
    for (const auto& value : c)
    {
        delegate(value);
    }
}

template<typename T>
void TensorPrinter::WriteToFile(const std::vector<T>& values)
{
    if (!m_OutputTensorFile.empty())
    {
        std::ofstream outputTensorFile;
        outputTensorFile.open(m_OutputTensorFile, std::ofstream::out | std::ofstream::trunc);
        if (outputTensorFile.is_open())
        {
            outputTensorFile << m_OutputBinding << ": ";
            std::copy(values.begin(), values.end(), std::ostream_iterator<T>(outputTensorFile, " "));
        }
        else
        {
            ARMNN_LOG(info) << "Output Tensor File: " << m_OutputTensorFile << " could not be opened!";
        }
        outputTensorFile.close();
    }
}

void PopulateTensorWithData(armnnUtils::TContainer& tensorData,
                            unsigned int numElements,
                            const std::string& dataTypeStr,
                            const armnn::Optional<QuantizationParams>& qParams,
                            const armnn::Optional<std::string>& dataFile)
{
    const bool readFromFile = dataFile.has_value() && !dataFile.value().empty();
    const bool quantizeData = qParams.has_value();

    std::ifstream inputTensorFile;
    if (readFromFile)
    {
        inputTensorFile = std::ifstream(dataFile.value());
    }

    if (dataTypeStr.compare("float") == 0)
    {
        if (quantizeData)
        {
            const float qScale  = qParams.value().first;
            const int   qOffset = qParams.value().second;

            tensorData = readFromFile ?
                         ParseDataArray<armnn::DataType::QAsymmU8>(inputTensorFile, qScale, qOffset) :
                         GenerateDummyTensorData<armnn::DataType::QAsymmU8>(numElements);
        }
        else
        {
            tensorData = readFromFile ?
                         ParseDataArray<armnn::DataType::Float32>(inputTensorFile) :
                         GenerateDummyTensorData<armnn::DataType::Float32>(numElements);
        }
    }
    else if (dataTypeStr.compare("int") == 0)
    {
        tensorData = readFromFile ?
                     ParseDataArray<armnn::DataType::Signed32>(inputTensorFile) :
                     GenerateDummyTensorData<armnn::DataType::Signed32>(numElements);
    }
    else if (dataTypeStr.compare("qsymms8") == 0)
    {
        tensorData = readFromFile ?
                     ParseDataArray<armnn::DataType::QSymmS8>(inputTensorFile) :
                     GenerateDummyTensorData<armnn::DataType::QSymmS8>(numElements);
    }
    else if (dataTypeStr.compare("qasymm8") == 0 || dataTypeStr.compare("qasymmu8") == 0)
    {
        tensorData = readFromFile ?
                     ParseDataArray<armnn::DataType::QAsymmU8>(inputTensorFile) :
                     GenerateDummyTensorData<armnn::DataType::QAsymmU8>(numElements);
    }
    else if (dataTypeStr.compare("qasymms8") == 0)
    {
        tensorData = readFromFile ?
                     ParseDataArray<armnn::DataType::QAsymmS8>(inputTensorFile) :
                     GenerateDummyTensorData<armnn::DataType::QAsymmS8>(numElements);
    }
    else
    {
        std::string errorMessage = "Unsupported tensor data type " + dataTypeStr;
        ARMNN_LOG(fatal) << errorMessage;

        inputTensorFile.close();
        throw armnn::Exception(errorMessage);
    }

    inputTensorFile.close();
}

bool ValidatePath(const std::string& file, const bool expectFile)
{
    if (!fs::exists(file))
    {
        std::cerr << "Given file path '" << file << "' does not exist" << std::endl;
        return false;
    }
    if (!fs::is_regular_file(file) && expectFile)
    {
        std::cerr << "Given file path '" << file << "' is not a regular file" << std::endl;
        return false;
    }
    return true;
}

bool ValidatePaths(const std::vector<std::string>& fileVec, const bool expectFile)
{
    bool allPathsValid = true;
    for (auto const& file : fileVec)
    {
        if(!ValidatePath(file, expectFile))
        {
            allPathsValid = false;
        }
    }
    return allPathsValid;
}



