//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/BackendRegistry.hpp>      // for BackendRegistryInstance
#include <armnn/Logging.hpp>              // for ScopedRecord, ARMNN_LOG
#include <armnn/utility/NumericCast.hpp>  // for numeric_cast
#include <armnn/utility/StringUtils.hpp>  // for StringTokenizer
#include <armnn/BackendId.hpp>            // for BackendId, BackendIdSet
#include <armnn/Optional.hpp>             // for Optional, EmptyOptional
#include <armnn/Tensor.hpp>               // for Tensor, TensorInfo
#include <armnn/TypesUtils.hpp>           // for Dequantize
#include <chrono>                         // for duration
#include <functional>                     // for function
#include <fstream>
#include <iomanip>
#include <iostream>                       // for ofstream, basic_istream
#include <ratio>                          // for milli
#include <string>                         // for string, getline, basic_string
#include <type_traits>                    // for enable_if_t, is_floating_point
#include <unordered_set>                  // for operator!=, operator==, _No...
#include <vector>                         // for vector
#include <math.h>                         // for pow, sqrt
#include <stdint.h>                       // for int32_t
#include <stdio.h>                        // for printf, size_t
#include <stdlib.h>                       // for abs
#include <algorithm>                      // for find, for_each

/**
 * Given a measured duration and a threshold time tell the user whether we succeeded or not.
 *
 * @param duration the measured inference duration.
 * @param thresholdTime the threshold time in milliseconds.
 * @return false if the measured time exceeded the threshold.
 */
bool CheckInferenceTimeThreshold(const std::chrono::duration<double, std::milli>& duration,
                                 const double& thresholdTime);

inline bool CheckRequestedBackendsAreValid(const std::vector<armnn::BackendId>& backendIds,
                                           armnn::Optional<std::string&> invalidBackendIds = armnn::EmptyOptional())
{
    if (backendIds.empty())
    {
        return false;
    }

    armnn::BackendIdSet validBackendIds = armnn::BackendRegistryInstance().GetBackendIds();

    bool allValid = true;
    for (const auto& backendId : backendIds)
    {
        if (std::find(validBackendIds.begin(), validBackendIds.end(), backendId) == validBackendIds.end())
        {
            allValid = false;
            if (invalidBackendIds)
            {
                if (!invalidBackendIds.value().empty())
                {
                    invalidBackendIds.value() += ", ";
                }
                invalidBackendIds.value() += backendId;
            }
        }
    }
    return allValid;
}

std::vector<unsigned int> ParseArray(std::istream& stream);

/// Splits a given string at every accurance of delimiter into a vector of string
std::vector<std::string> ParseStringList(const std::string& inputString, const char* delimiter);

double ComputeByteLevelRMSE(const void* expected, const void* actual, const size_t size);

/// Dequantize an array of a given type
/// @param array Type erased array to dequantize
/// @param numElements Elements in the array
/// @param array Type erased array to dequantize
template <typename T>
std::vector<float> DequantizeArray(const void* array, unsigned int numElements, float scale, int32_t offset)
{
    const T* quantizedArray = reinterpret_cast<const T*>(array);
    std::vector<float> dequantizedVector;
    dequantizedVector.reserve(numElements);
    for (unsigned int i = 0; i < numElements; ++i)
    {
        float f = armnn::Dequantize(*(quantizedArray + i), scale, offset);
        dequantizedVector.push_back(f);
    }
    return dequantizedVector;
}

void LogAndThrow(std::string eMsg);

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

/// Returns a function of read the given type as a string
template <typename Integer, typename std::enable_if_t<std::is_integral<Integer>::value>* = nullptr>
std::function<Integer(const std::string&)> GetParseElementFunc()
{
    return [](const std::string& s) { return armnn::numeric_cast<Integer>(std::stoi(s)); };
}

template <typename Float, std::enable_if_t<std::is_floating_point<Float>::value>* = nullptr>
std::function<Float(const std::string&)> GetParseElementFunc()
{
    return [](const std::string& s) { return std::stof(s); };
}

template <typename T>
void PopulateTensorWithData(T* tensor,
                            const unsigned int numElements,
                            const armnn::Optional<std::string>& dataFile,
                            const std::string& inputName)
{
    const bool readFromFile = dataFile.has_value() && !dataFile.value().empty();

    std::ifstream inputTensorFile;
    if (!readFromFile)
    {
        std::fill(tensor, tensor + numElements, 0);
        return;
    }
    else
    {
        inputTensorFile = std::ifstream(dataFile.value());
    }

    auto parseElementFunc = GetParseElementFunc<T>();
    std::string line;
    unsigned int index = 0;
    while (std::getline(inputTensorFile, line))
    {
        std::vector<std::string> tokens = armnn::stringUtils::StringTokenizer(line, "\t ,:");
        for (const std::string& token : tokens)
        {
            if (!token.empty()) // See https://stackoverflow.com/questions/10437406/
            {
                try
                {
                    if (index == numElements)
                    {
                        ARMNN_LOG(error) << "Number of elements: " << (index +1) << " in file \"" << dataFile.value()
                                         << "\" does not match number of elements: " << numElements
                                         << " for input \"" << inputName << "\".";
                    }
                    *(tensor + index) = parseElementFunc(token);
                    index++;
                }
                catch (const std::exception&)
                {
                    ARMNN_LOG(error) << "'" << token << "' is not a valid number. It has been ignored.";
                }
            }
        }
    }

    if (index != numElements)
    {
        ARMNN_LOG(error) << "Number of elements: " << (index +1) << " in file \"" << inputName
                         << "\" does not match number of elements: " << numElements
                         << " for input \"" << inputName << "\".";
    }
}

template<typename T>
void WriteToFile(const std::string& outputTensorFileName,
                 const std::string& outputName,
                 const T* const array,
                 const unsigned int numElements)
{
    std::ofstream outputTensorFile;
    outputTensorFile.open(outputTensorFileName, std::ofstream::out | std::ofstream::trunc);
    if (outputTensorFile.is_open())
    {
        outputTensorFile << outputName << ": ";
        for (std::size_t i = 0; i < numElements; ++i)
        {
            outputTensorFile << +array[i] << " ";
        }
    }
    else
    {
        ARMNN_LOG(info) << "Output Tensor File: " << outputTensorFileName << " could not be opened!";
    }
    outputTensorFile.close();
}

struct OutputWriteInfo
{
    const armnn::Optional<std::string>& m_OutputTensorFile;
    const std::string& m_OutputName;
    const armnn::Tensor& m_Tensor;
    const bool m_PrintTensor;
};

template <typename T>
void PrintTensor(OutputWriteInfo& info, const char* formatString)
{
    const T* array = reinterpret_cast<const T*>(info.m_Tensor.GetMemoryArea());

    if (info.m_OutputTensorFile.has_value())
    {
        WriteToFile(info.m_OutputTensorFile.value(),
                    info.m_OutputName,
                    array,
                    info.m_Tensor.GetNumElements());
    }

    if (info.m_PrintTensor)
    {
        for (unsigned int i = 0; i < info.m_Tensor.GetNumElements(); i++)
        {
            printf(formatString, array[i]);
        }
    }
}

template <typename T>
void PrintQuantizedTensor(OutputWriteInfo& info)
{
    std::vector<float> dequantizedValues;
    auto tensor = info.m_Tensor;
    dequantizedValues = DequantizeArray<T>(tensor.GetMemoryArea(),
                                           tensor.GetNumElements(),
                                           tensor.GetInfo().GetQuantizationScale(),
                                           tensor.GetInfo().GetQuantizationOffset());

    if (info.m_OutputTensorFile.has_value())
    {
        WriteToFile(info.m_OutputTensorFile.value(),
                    info.m_OutputName,
                    dequantizedValues.data(),
                    tensor.GetNumElements());
    }

    if (info.m_PrintTensor)
    {
        std::for_each(dequantizedValues.begin(), dequantizedValues.end(), [&](float value)
        {
            printf("%f ", value);
        });
    }
}

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
