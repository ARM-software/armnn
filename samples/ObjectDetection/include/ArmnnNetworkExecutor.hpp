//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "Types.hpp"

#include "armnn/ArmNN.hpp"
#include "armnnTfLiteParser/ITfLiteParser.hpp"
#include "armnnUtils/DataLayoutIndexed.hpp"
#include <armnn/Logging.hpp>

#include <string>
#include <vector>

namespace od
{
/**
* @brief Used to load in a network through ArmNN and run inference on it against a given backend.
*
*/
class ArmnnNetworkExecutor
{
private:
    armnn::IRuntimePtr m_Runtime;
    armnn::NetworkId m_NetId{};
    mutable InferenceResults m_OutputBuffer;
    armnn::InputTensors     m_InputTensors;
    armnn::OutputTensors    m_OutputTensors;
    std::vector<armnnTfLiteParser::BindingPointInfo> m_outputBindingInfo;

    std::vector<std::string> m_outputLayerNamesList;

    armnnTfLiteParser::BindingPointInfo m_inputBindingInfo;

    void PrepareTensors(const void* inputData, const size_t dataBytes);

    template <typename Enumeration>
    auto log_as_int(Enumeration value)
    -> typename std::underlying_type<Enumeration>::type
    {
        return static_cast<typename std::underlying_type<Enumeration>::type>(value);
    }

public:
    ArmnnNetworkExecutor() = delete;

    /**
    * @brief Initializes the network with the given input data. Parsed through TfLiteParser and optimized for a
    *        given backend.
    *
    * Note that the output layers names order in m_outputLayerNamesList affects the order of the feature vectors
    * in output of the Run method.
    *
    *       * @param[in] modelPath - Relative path to the model file
    *       * @param[in] backends - The list of preferred backends to run inference on
    */
    ArmnnNetworkExecutor(std::string& modelPath,
                         std::vector<armnn::BackendId>& backends);

    /**
    * @brief Returns the aspect ratio of the associated model in the order of width, height.
    */
    Size GetImageAspectRatio();

    armnn::DataType GetInputDataType() const;

    /**
    * @brief Runs inference on the provided input data, and stores the results in the provided InferenceResults object.
    *
    * @param[in] inputData - input frame data
    * @param[in] dataBytes - input data size in bytes
    * @param[out] results - Vector of DetectionResult objects used to store the output result.
    */
    bool Run(const void* inputData, const size_t dataBytes, InferenceResults& outResults);

};
}// namespace od