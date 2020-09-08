//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "LstmCommon.hpp"

#include <memory>
#include <string>
#include <vector>

#include <armnn/TypesUtils.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <array>
#include <string>

#include "InferenceTestImage.hpp"

namespace
{

template<typename T, typename TParseElementFunc>
std::vector<T> ParseArrayImpl(std::istream& stream, TParseElementFunc parseElementFunc, const char * chars = "\t ,:")
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

template<armnn::DataType NonQuantizedType>
auto ParseDataArray(std::istream & stream);

template<armnn::DataType QuantizedType>
auto ParseDataArray(std::istream& stream,
                    const float& quantizationScale,
                    const int32_t& quantizationOffset);

// NOTE: declaring the template specialisations inline to prevent them
//       being flagged as unused functions when -Werror=unused-function is in effect
template<>
inline auto ParseDataArray<armnn::DataType::Float32>(std::istream & stream)
{
    return ParseArrayImpl<float>(stream, [](const std::string& s) { return std::stof(s); });
}

template<>
inline auto ParseDataArray<armnn::DataType::Signed32>(std::istream & stream)
{
    return ParseArrayImpl<int>(stream, [](const std::string & s) { return std::stoi(s); });
}

template<>
inline auto ParseDataArray<armnn::DataType::QAsymmU8>(std::istream& stream,
                                                      const float& quantizationScale,
                                                      const int32_t& quantizationOffset)
{
    return ParseArrayImpl<uint8_t>(stream,
                                   [&quantizationScale, &quantizationOffset](const std::string & s)
                                   {
                                       return armnn::numeric_cast<uint8_t>(
                                               armnn::Quantize<uint8_t>(std::stof(s),
                                                                         quantizationScale,
                                                                         quantizationOffset));
                                   });
}

struct DeepSpeechV1TestCaseData
{
    DeepSpeechV1TestCaseData(
        const LstmInput& inputData,
        const LstmInput& expectedOutputData)
        : m_InputData(inputData)
        , m_ExpectedOutputData(expectedOutputData)
    {}

    LstmInput m_InputData;
    LstmInput m_ExpectedOutputData;
};

class DeepSpeechV1Database
{
public:
    explicit DeepSpeechV1Database(const std::string& inputSeqDir, const std::string& prevStateHDir,
                                  const std::string& prevStateCDir, const std::string& logitsDir,
                                  const std::string& newStateHDir, const std::string& newStateCDir);

    std::unique_ptr<DeepSpeechV1TestCaseData> GetTestCaseData(unsigned int testCaseId);

private:
    std::string m_InputSeqDir;
    std::string m_PrevStateHDir;
    std::string m_PrevStateCDir;
    std::string m_LogitsDir;
    std::string m_NewStateHDir;
    std::string m_NewStateCDir;
};

DeepSpeechV1Database::DeepSpeechV1Database(const std::string& inputSeqDir, const std::string& prevStateHDir,
                                           const std::string& prevStateCDir, const std::string& logitsDir,
                                           const std::string& newStateHDir, const std::string& newStateCDir)
    : m_InputSeqDir(inputSeqDir)
    , m_PrevStateHDir(prevStateHDir)
    , m_PrevStateCDir(prevStateCDir)
    , m_LogitsDir(logitsDir)
    , m_NewStateHDir(newStateHDir)
    , m_NewStateCDir(newStateCDir)
{}

std::unique_ptr<DeepSpeechV1TestCaseData> DeepSpeechV1Database::GetTestCaseData(unsigned int testCaseId)
{
    // Load test case input
    const std::string inputSeqPath   = m_InputSeqDir + "input_node_0_flat.txt";
    const std::string prevStateCPath = m_PrevStateCDir + "previous_state_c_0.txt";
    const std::string prevStateHPath = m_PrevStateHDir + "previous_state_h_0.txt";

    std::vector<float> inputSeqData;
    std::vector<float> prevStateCData;
    std::vector<float> prevStateHData;

    std::ifstream inputSeqFile(inputSeqPath);
    std::ifstream prevStateCTensorFile(prevStateCPath);
    std::ifstream prevStateHTensorFile(prevStateHPath);

    try
    {
        inputSeqData   = ParseDataArray<armnn::DataType::Float32>(inputSeqFile);
        prevStateCData = ParseDataArray<armnn::DataType::Float32>(prevStateCTensorFile);
        prevStateHData = ParseDataArray<armnn::DataType::Float32>(prevStateHTensorFile);
    }
    catch (const InferenceTestImageException& e)
    {
        ARMNN_LOG(fatal) << "Failed to load image for test case " << testCaseId << ". Error: " << e.what();
        return nullptr;
    }

    // Prepare test case expected output
    const std::string logitsPath   = m_LogitsDir + "logits.txt";
    const std::string newStateCPath = m_NewStateCDir + "new_state_c.txt";
    const std::string newStateHPath = m_NewStateHDir + "new_state_h.txt";

    std::vector<float> logitsData;
    std::vector<float> expectedNewStateCData;
    std::vector<float> expectedNewStateHData;

    std::ifstream logitsTensorFile(logitsPath);
    std::ifstream newStateCTensorFile(newStateCPath);
    std::ifstream newStateHTensorFile(newStateHPath);

    try
    {
        logitsData     = ParseDataArray<armnn::DataType::Float32>(logitsTensorFile);
        expectedNewStateCData = ParseDataArray<armnn::DataType::Float32>(newStateCTensorFile);
        expectedNewStateHData = ParseDataArray<armnn::DataType::Float32>(newStateHTensorFile);
    }
    catch (const InferenceTestImageException& e)
    {
        ARMNN_LOG(fatal) << "Failed to load image for test case " << testCaseId << ". Error: " << e.what();
        return nullptr;
    }

    // use the struct for representing input and output data
    LstmInput inputDataSingleTest(inputSeqData, prevStateHData, prevStateCData);

    LstmInput expectedOutputsSingleTest(logitsData, expectedNewStateHData, expectedNewStateCData);

    return std::make_unique<DeepSpeechV1TestCaseData>(inputDataSingleTest, expectedOutputsSingleTest);
}

} // anonymous namespace
