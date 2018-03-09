//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#pragma once

#include "armnn/ArmNN.hpp"

#include <boost/log/trivial.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>

#include <map>
#include <string>

template<typename TContainer>
inline armnn::InputTensors MakeInputTensors(const std::pair<armnn::LayerBindingId, armnn::TensorInfo>& input,
    const TContainer& inputTensorData)
{
    if (inputTensorData.size() != input.second.GetNumElements())
    {
        throw armnn::Exception(boost::str(boost::format("Input tensor has incorrect size. Expected %1% elements "
            "but got %2%.") % input.second.GetNumElements() % inputTensorData.size()));
    }
    return { { input.first, armnn::ConstTensor(input.second, inputTensorData.data()) } };
}

template<typename TContainer>
inline armnn::OutputTensors MakeOutputTensors(const std::pair<armnn::LayerBindingId, armnn::TensorInfo>& output,
    TContainer& outputTensorData)
{
    if (outputTensorData.size() != output.second.GetNumElements())
    {
        throw armnn::Exception("Output tensor has incorrect size");
    }
    return { { output.first, armnn::Tensor(output.second, outputTensorData.data()) } };
}

template <typename IParser, typename TDataType>
class InferenceModel
{
public:
    using DataType = TDataType;

    struct CommandLineOptions
    {
        std::string m_ModelDir;
        armnn::Compute m_ComputeDevice;
    };

    static void AddCommandLineOptions(boost::program_options::options_description& desc, CommandLineOptions& options)
    {
        namespace po = boost::program_options;

        desc.add_options()
            ("model-dir,m", po::value<std::string>(&options.m_ModelDir)->required(),
                "Path to directory containing model files (.caffemodel/.prototxt)")
            ("compute,c", po::value<armnn::Compute>(&options.m_ComputeDevice)->default_value(armnn::Compute::CpuAcc),
                "Which device to run layers on by default. Possible choices: CpuAcc, CpuRef, GpuAcc");
    }

    struct Params
    {
        std::string m_ModelPath;
        std::string m_InputBinding;
        std::string m_OutputBinding;
        const armnn::TensorShape* m_InputTensorShape;
        armnn::Compute m_ComputeDevice;
        bool m_IsModelBinary;

        Params()
         : m_InputTensorShape(nullptr)
         , m_ComputeDevice(armnn::Compute::CpuRef)
         , m_IsModelBinary(true)
        {
        }
    };


    InferenceModel(const Params& params)
     : m_Runtime(armnn::IRuntime::Create(params.m_ComputeDevice))
    {
        const std::string& modelPath = params.m_ModelPath;

        // Create a network from a file on disk
        auto parser(IParser::Create());

        std::map<std::string, armnn::TensorShape> inputShapes;
        if (params.m_InputTensorShape)
        {
            inputShapes[params.m_InputBinding] = *params.m_InputTensorShape;
        }
        std::vector<std::string> requestedOutputs{ params.m_OutputBinding };

        // Handle text and binary input differently by calling the corresponding parser function
        armnn::INetworkPtr network = (params.m_IsModelBinary ?
          parser->CreateNetworkFromBinaryFile(modelPath.c_str(), inputShapes, requestedOutputs) :
          parser->CreateNetworkFromTextFile(modelPath.c_str(), inputShapes, requestedOutputs));

        m_InputBindingInfo  = parser->GetNetworkInputBindingInfo(params.m_InputBinding);
        m_OutputBindingInfo = parser->GetNetworkOutputBindingInfo(params.m_OutputBinding);

        armnn::IOptimizedNetworkPtr optNet =
            armnn::Optimize(*network, m_Runtime->GetDeviceSpec());

        // Load the network into the runtime.
        armnn::Status ret = m_Runtime->LoadNetwork(m_NetworkIdentifier, std::move(optNet));
        if (ret == armnn::Status::Failure)
        {
            throw armnn::Exception("IRuntime::LoadNetwork failed");
        }
    }

    unsigned int GetOutputSize() const
    {
        return m_OutputBindingInfo.second.GetNumElements();
    }

    void Run(const std::vector<TDataType>& input, std::vector<TDataType>& output)
    {
        BOOST_ASSERT(output.size() == GetOutputSize());
        armnn::Status ret = m_Runtime->EnqueueWorkload(m_NetworkIdentifier,
                                                            MakeInputTensors(input),
                                                            MakeOutputTensors(output));
        if (ret == armnn::Status::Failure)
        {
            throw armnn::Exception("IRuntime::EnqueueWorkload failed");
        }
    }

private:
    template<typename TContainer>
    armnn::InputTensors MakeInputTensors(const TContainer& inputTensorData)
    {
        return ::MakeInputTensors(m_InputBindingInfo, inputTensorData);
    }

    template<typename TContainer>
    armnn::OutputTensors MakeOutputTensors(TContainer& outputTensorData)
    {
        return ::MakeOutputTensors(m_OutputBindingInfo, outputTensorData);
    }

    armnn::NetworkId m_NetworkIdentifier;
    armnn::IRuntimePtr m_Runtime;

    std::pair<armnn::LayerBindingId, armnn::TensorInfo> m_InputBindingInfo;
    std::pair<armnn::LayerBindingId, armnn::TensorInfo> m_OutputBindingInfo;
};
