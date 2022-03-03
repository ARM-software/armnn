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
#include "Profiling.hpp"

#include <string>
#include <vector>

namespace common
{
/**
* @brief Used to load in a network through ArmNN and run inference on it against a given backend.
*
*/
template <typename Tout>
class ArmnnNetworkExecutor
{
private:
    armnn::IRuntimePtr m_Runtime;
    armnn::NetworkId m_NetId{};
    mutable InferenceResults<Tout> m_OutputBuffer;
    armnn::InputTensors     m_InputTensors;
    armnn::OutputTensors    m_OutputTensors;
    std::vector<armnnTfLiteParser::BindingPointInfo> m_outputBindingInfo;
    Profiling m_profiling;
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
                         std::vector<armnn::BackendId>& backends,
                         bool isProfilingEnabled = false);

    /**
    * @brief Returns the aspect ratio of the associated model in the order of width, height.
    */
    Size GetImageAspectRatio();

    armnn::DataType GetInputDataType() const;

    float GetQuantizationScale();

    int GetQuantizationOffset();

    float GetOutputQuantizationScale(int tensorIndex);

    int GetOutputQuantizationOffset(int tensorIndex);

    /**
    * @brief Runs inference on the provided input data, and stores the results in the provided InferenceResults object.
    *
    * @param[in] inputData - input frame data
    * @param[in] dataBytes - input data size in bytes
    * @param[out] results - Vector of DetectionResult objects used to store the output result.
    */
    bool Run(const void* inputData, const size_t dataBytes, common::InferenceResults<Tout>& outResults);

};

template <typename Tout>
ArmnnNetworkExecutor<Tout>::ArmnnNetworkExecutor(std::string& modelPath,
                                           std::vector<armnn::BackendId>& preferredBackends,
                                           bool isProfilingEnabled):
        m_profiling(isProfilingEnabled),
        m_Runtime(armnn::IRuntime::Create(armnn::IRuntime::CreationOptions()))
{
    // Import the TensorFlow lite model.
    m_profiling.ProfilingStart();
    armnnTfLiteParser::ITfLiteParserPtr parser = armnnTfLiteParser::ITfLiteParser::Create();
    armnn::INetworkPtr network = parser->CreateNetworkFromBinaryFile(modelPath.c_str());

    std::vector<std::string> inputNames = parser->GetSubgraphInputTensorNames(0);

    m_inputBindingInfo = parser->GetNetworkInputBindingInfo(0, inputNames[0]);

    m_outputLayerNamesList = parser->GetSubgraphOutputTensorNames(0);

    std::vector<armnn::BindingPointInfo> outputBindings;
    for(const std::string& name : m_outputLayerNamesList)
    {
        m_outputBindingInfo.push_back(std::move(parser->GetNetworkOutputBindingInfo(0, name)));
    }
    std::vector<std::string> errorMessages;
    // optimize the network.
    armnn::IOptimizedNetworkPtr optNet = Optimize(*network,
                                                  preferredBackends,
                                                  m_Runtime->GetDeviceSpec(),
                                                  armnn::OptimizerOptions(),
                                                  armnn::Optional<std::vector<std::string>&>(errorMessages));

    if (!optNet)
    {
        const std::string errorMessage{"ArmnnNetworkExecutor: Failed to optimize network"};
        ARMNN_LOG(error) << errorMessage;
        throw armnn::Exception(errorMessage);
    }

    // Load the optimized network onto the m_Runtime device
    std::string errorMessage;
    if (armnn::Status::Success != m_Runtime->LoadNetwork(m_NetId, std::move(optNet), errorMessage))
    {
        ARMNN_LOG(error) << errorMessage;
        throw armnn::Exception(errorMessage);
    }

    //pre-allocate memory for output (the size of it never changes)
    for (int it = 0; it < m_outputLayerNamesList.size(); ++it)
    {
        const armnn::DataType dataType = m_outputBindingInfo[it].second.GetDataType();
        const armnn::TensorShape& tensorShape = m_outputBindingInfo[it].second.GetShape();

        std::vector<Tout> oneLayerOutResult;
        oneLayerOutResult.resize(tensorShape.GetNumElements(), 0);
        m_OutputBuffer.emplace_back(oneLayerOutResult);

        // Make ArmNN output tensors
        m_OutputTensors.reserve(m_OutputBuffer.size());
        for (size_t it = 0; it < m_OutputBuffer.size(); ++it)
        {
            m_OutputTensors.emplace_back(std::make_pair(
                    m_outputBindingInfo[it].first,
                    armnn::Tensor(m_outputBindingInfo[it].second,
                                  m_OutputBuffer.at(it).data())
            ));
        }
    }
    m_profiling.ProfilingStopAndPrintUs("ArmnnNetworkExecutor time");
}

template <typename Tout>
armnn::DataType ArmnnNetworkExecutor<Tout>::GetInputDataType() const
{
    return m_inputBindingInfo.second.GetDataType();
}

template <typename Tout>
void ArmnnNetworkExecutor<Tout>::PrepareTensors(const void* inputData, const size_t dataBytes)
{
    assert(m_inputBindingInfo.second.GetNumBytes() >= dataBytes);
    m_InputTensors.clear();
    m_InputTensors = {{ m_inputBindingInfo.first, armnn::ConstTensor(m_inputBindingInfo.second, inputData)}};
}

template <typename Tout>
bool ArmnnNetworkExecutor<Tout>::Run(const void* inputData, const size_t dataBytes, InferenceResults<Tout>& outResults)
{
    m_profiling.ProfilingStart();
    /* Prepare tensors if they are not ready */
    ARMNN_LOG(debug) << "Preparing tensors...";
    this->PrepareTensors(inputData, dataBytes);
    ARMNN_LOG(trace) << "Running inference...";

    armnn::Status ret = m_Runtime->EnqueueWorkload(m_NetId, m_InputTensors, m_OutputTensors);

    std::stringstream inferenceFinished;
    inferenceFinished << "Inference finished with code {" << log_as_int(ret) << "}\n";

    ARMNN_LOG(trace) << inferenceFinished.str();

    if (ret == armnn::Status::Failure)
    {
        ARMNN_LOG(error) << "Failed to perform inference.";
    }

    outResults.reserve(m_outputLayerNamesList.size());
    outResults = m_OutputBuffer;
    m_profiling.ProfilingStopAndPrintUs("Total inference time");
    return (armnn::Status::Success == ret);
}

template <typename Tout>
float ArmnnNetworkExecutor<Tout>::GetQuantizationScale()
{
    return this->m_inputBindingInfo.second.GetQuantizationScale();
}

template <typename Tout>
int ArmnnNetworkExecutor<Tout>::GetQuantizationOffset()
{
    return this->m_inputBindingInfo.second.GetQuantizationOffset();
}

template <typename Tout>
float ArmnnNetworkExecutor<Tout>::GetOutputQuantizationScale(int tensorIndex)
{
    assert(this->m_outputLayerNamesList.size() > tensorIndex);
    return this->m_outputBindingInfo[tensorIndex].second.GetQuantizationScale();
}

template <typename Tout>
int ArmnnNetworkExecutor<Tout>::GetOutputQuantizationOffset(int tensorIndex)
{
    assert(this->m_outputLayerNamesList.size() > tensorIndex);
    return this->m_outputBindingInfo[tensorIndex].second.GetQuantizationOffset();
}

template <typename Tout>
Size ArmnnNetworkExecutor<Tout>::GetImageAspectRatio()
{
    const auto shape = m_inputBindingInfo.second.GetShape();
    assert(shape.GetNumDimensions() == 4);
    armnnUtils::DataLayoutIndexed nhwc(armnn::DataLayout::NHWC);
    return Size(shape[nhwc.GetWidthIndex()],
                shape[nhwc.GetHeightIndex()]);
}
}// namespace common