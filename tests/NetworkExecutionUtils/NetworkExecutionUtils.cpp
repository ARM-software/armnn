//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NetworkExecutionUtils.hpp"

#include <armnnUtils/Filesystem.hpp>
#include <iterator>
std::vector<std::string> ParseStringList(const std::string& inputString, const char* delimiter)
{
    std::stringstream stream(inputString);
    return ParseArrayImpl<std::string>(stream, [](const std::string& s) {
        return armnn::stringUtils::StringTrimCopy(s); }, delimiter);
}

bool CheckInferenceTimeThreshold(const std::chrono::duration<double, std::milli>& duration,
                                 const double& thresholdTime)
{
    ARMNN_LOG(info) << "Inference time: " << std::setprecision(2)
                    << std::fixed << duration.count() << " ms\n";
    // If thresholdTime == 0.0 (default), then it hasn't been supplied at command line
    if (thresholdTime != 0.0)
    {
        ARMNN_LOG(info) << "Threshold time: " << std::setprecision(2)
                        << std::fixed << thresholdTime << " ms";
        auto thresholdMinusInference = thresholdTime - duration.count();
        ARMNN_LOG(info) << "Threshold time - Inference time: " << std::setprecision(2)
                        << std::fixed << thresholdMinusInference << " ms" << "\n";
        if (thresholdMinusInference < 0)
        {
            std::string errorMessage = "Elapsed inference time is greater than provided threshold time.";
            ARMNN_LOG(fatal) << errorMessage;
            return false;
        }
    }
    return true;
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

std::vector<unsigned int> ParseArray(std::istream& stream)
{
    return ParseArrayImpl<unsigned int>(
            stream,
            [](const std::string& s) { return armnn::numeric_cast<unsigned int>(std::stoi(s)); });
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

void LogAndThrow(std::string eMsg)
{
    ARMNN_LOG(error) << eMsg;
    throw armnn::Exception(eMsg);
}

/// Compute the root-mean-square error (RMSE) at a byte level between two tensors of the same size.
/// @param expected
/// @param actual
/// @param size size of the tensor in bytes.
/// @return float the RMSE
double ComputeByteLevelRMSE(const void* expected, const void* actual, const size_t size)
{
    const uint8_t* byteExpected = reinterpret_cast<const uint8_t*>(expected);
    const uint8_t* byteActual = reinterpret_cast<const uint8_t*>(actual);

    double errorSum = 0;
    for (unsigned int i = 0; i < size; i++)
    {
        int difference = byteExpected[i] - byteActual[i];
        errorSum += std::pow(difference, 2);
    }
    return std::sqrt(errorSum/armnn::numeric_cast<double>(size));
}
