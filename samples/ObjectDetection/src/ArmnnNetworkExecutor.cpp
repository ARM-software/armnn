//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ArmnnNetworkExecutor.hpp"
#include "Types.hpp"

#include <random>
#include <string>

namespace od
{

armnn::DataType ArmnnNetworkExecutor::GetInputDataType() const
{
    return m_inputBindingInfo.second.GetDataType();
}

ArmnnNetworkExecutor::ArmnnNetworkExecutor(std::string& modelPath,
                                           std::vector<armnn::BackendId>& preferredBackends)
: m_Runtime(armnn::IRuntime::Create(armnn::IRuntime::CreationOptions()))
{
    // Import the TensorFlow lite model.
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
    }

    //pre-allocate memory for output (the size of it never changes)
    for (int it = 0; it < m_outputLayerNamesList.size(); ++it)
    {
        const armnn::DataType dataType = m_outputBindingInfo[it].second.GetDataType();
        const armnn::TensorShape& tensorShape = m_outputBindingInfo[it].second.GetShape();

        InferenceResult oneLayerOutResult;
        switch (dataType)
        {
            case armnn::DataType::Float32:
            {
                oneLayerOutResult.resize(tensorShape.GetNumElements(), 0);
                break;
            }
            default:
            {
                errorMessage = "ArmnnNetworkExecutor: unsupported output tensor data type";
                ARMNN_LOG(error) << errorMessage << " " << log_as_int(dataType);
                throw armnn::Exception(errorMessage);
            }
        }

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

}

void ArmnnNetworkExecutor::PrepareTensors(const void* inputData, const size_t dataBytes)
{
    assert(m_inputBindingInfo.second.GetNumBytes() >= dataBytes);
    m_InputTensors.clear();
    m_InputTensors = {{ m_inputBindingInfo.first, armnn::ConstTensor(m_inputBindingInfo.second, inputData)}};
}

bool ArmnnNetworkExecutor::Run(const void* inputData, const size_t dataBytes, InferenceResults& outResults)
{
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

    return (armnn::Status::Success == ret);
}

Size ArmnnNetworkExecutor::GetImageAspectRatio()
{
    const auto shape = m_inputBindingInfo.second.GetShape();
    assert(shape.GetNumDimensions() == 4);
    armnnUtils::DataLayoutIndexed nhwc(armnn::DataLayout::NHWC);
    return Size(shape[nhwc.GetWidthIndex()],
                shape[nhwc.GetHeightIndex()]);
}
}// namespace od