//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/IRuntime.hpp>
#include <armnn/Types.hpp>
#include <armnn/Logging.hpp>
#include <armnn/utility/StringUtils.hpp>
#include <armnnUtils/TContainer.hpp>

#include <iostream>
#include <fstream>


std::vector<unsigned int> ParseArray(std::istream& stream);

/// Splits a given string at every accurance of delimiter into a vector of string
std::vector<std::string> ParseStringList(const std::string& inputString, const char* delimiter);

struct TensorPrinter
{
    TensorPrinter(const std::string& binding,
                  const armnn::TensorInfo& info,
                  const std::string& outputTensorFile,
                  bool dequantizeOutput,
                  bool printToConsole = true);

    void operator()(const std::vector<float>& values);

    void operator()(const std::vector<uint8_t>& values);

    void operator()(const std::vector<int>& values);

    void operator()(const std::vector<int8_t>& values);

private:
    template<typename Container, typename Delegate>
    void ForEachValue(const Container& c, Delegate delegate);

    template<typename T>
    void WriteToFile(const std::vector<T>& values);

    std::string m_OutputBinding;
    float m_Scale;
    int m_Offset;
    std::string m_OutputTensorFile;
    bool m_DequantizeOutput;
    bool m_PrintToConsole;
};

using QuantizationParams = std::pair<float, int32_t>;

void PopulateTensorWithData(armnnUtils::TContainer& tensorData,
                            unsigned int numElements,
                            const std::string& dataTypeStr,
                            const armnn::Optional<QuantizationParams>& qParams,
                            const armnn::Optional<std::string>& dataFile);

/**
 * Verifies if the given string is a valid path. Reports invalid paths to std::err.
 * @param file string - A string containing the path to check
 * @param expectFile bool - If true, checks for a regular file.
 * @return bool - True if given string is a valid path., false otherwise.
 * */
bool ValidatePath(const std::string& file, const bool expectFile);

/**
 * Verifies if a given vector of strings are valid paths. Reports invalid paths to std::err.
 * @param fileVec vector of string - A vector of string containing the paths to check
 * @param expectFile bool - If true, checks for a regular file.
 * @return bool - True if all given strings are valid paths., false otherwise.
 * */
bool ValidatePaths(const std::vector<std::string>& fileVec, const bool expectFile);

template<typename T, typename TParseElementFunc>
std::vector<T> ParseArrayImpl(std::istream& stream, TParseElementFunc parseElementFunc, const char* chars = "\t ,:")
{
    std::vector<T> result;
    // Processes line-by-line.
    std::string line;
    while (std::getline(stream, line))
    {
        std::vector<std::string> tokens = armnn::stringUtils::StringTokenizer(line, chars);
        for (const std::string& token : tokens)
        {
            if (!token.empty()) // See https://stackoverflow.com/questions/10437406/
            {
                try
                {
                    result.push_back(parseElementFunc(token));
                }
                catch (const std::exception&)
                {
                    ARMNN_LOG(error) << "'" << token << "' is not a valid number. It has been ignored.";
                }
            }
        }
    }

    return result;
}

template <typename T, typename TParseElementFunc>
void PopulateTensorWithDataGeneric(std::vector<T>& tensorData,
                                   unsigned int numElements,
                                   const armnn::Optional<std::string>& dataFile,
                                   TParseElementFunc parseFunction)
{
    const bool readFromFile = dataFile.has_value() && !dataFile.value().empty();

    std::ifstream inputTensorFile;
    if (readFromFile)
    {
        inputTensorFile = std::ifstream(dataFile.value());
    }

    tensorData = readFromFile ?
                 ParseArrayImpl<T>(inputTensorFile, parseFunction) :
                 std::vector<T>(numElements, static_cast<T>(0));
}
