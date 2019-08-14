//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once
#include <armnn/ArmNN.hpp>

#if defined(ARMNN_SERIALIZER)
#include "armnnDeserializer/IDeserializer.hpp"
#endif
#if defined(ARMNN_TF_LITE_PARSER)
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#endif
#if defined(ARMNN_ONNX_PARSER)
#include <armnnOnnxParser/IOnnxParser.hpp>
#endif

#include <HeapProfiling.hpp>
#include <TensorIOUtils.hpp>

#include <backendsCommon/BackendRegistry.hpp>

#include <boost/algorithm/string/join.hpp>
#include <boost/exception/exception.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <boost/log/trivial.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/variant.hpp>

#include <algorithm>
#include <chrono>
#include <iterator>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <type_traits>

namespace
{

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

} // anonymous namespace

namespace InferenceModelInternal
{
using BindingPointInfo = armnn::BindingPointInfo;

using QuantizationParams = std::pair<float,int32_t>;

struct Params
{
    std::string                     m_ModelPath;
    std::vector<std::string>        m_InputBindings;
    std::vector<armnn::TensorShape> m_InputShapes;
    std::vector<std::string>        m_OutputBindings;
    std::vector<armnn::BackendId>   m_ComputeDevices;
    std::string                     m_DynamicBackendsPath;
    size_t                          m_SubgraphId;
    bool                            m_IsModelBinary;
    bool                            m_VisualizePostOptimizationModel;
    bool                            m_EnableFp16TurboMode;

    Params()
        : m_ComputeDevices{}
        , m_SubgraphId(0)
        , m_IsModelBinary(true)
        , m_VisualizePostOptimizationModel(false)
        , m_EnableFp16TurboMode(false)
    {}
};

} // namespace InferenceModelInternal

template <typename IParser>
struct CreateNetworkImpl
{
public:
    using Params = InferenceModelInternal::Params;

    static armnn::INetworkPtr Create(const Params& params,
                                     std::vector<armnn::BindingPointInfo>& inputBindings,
                                     std::vector<armnn::BindingPointInfo>& outputBindings)
    {
        const std::string& modelPath = params.m_ModelPath;

        // Create a network from a file on disk
        auto parser(IParser::Create());

        std::map<std::string, armnn::TensorShape> inputShapes;
        if (!params.m_InputShapes.empty())
        {
            const size_t numInputShapes   = params.m_InputShapes.size();
            const size_t numInputBindings = params.m_InputBindings.size();
            if (numInputShapes < numInputBindings)
            {
                throw armnn::Exception(boost::str(boost::format(
                    "Not every input has its tensor shape specified: expected=%1%, got=%2%")
                    % numInputBindings % numInputShapes));
            }

            for (size_t i = 0; i < numInputShapes; i++)
            {
                inputShapes[params.m_InputBindings[i]] = params.m_InputShapes[i];
            }
        }

        std::vector<std::string> requestedOutputs = params.m_OutputBindings;
        armnn::INetworkPtr network{nullptr, [](armnn::INetwork *){}};

        {
            ARMNN_SCOPED_HEAP_PROFILING("Parsing");
            // Handle text and binary input differently by calling the corresponding parser function
            network = (params.m_IsModelBinary ?
                parser->CreateNetworkFromBinaryFile(modelPath.c_str(), inputShapes, requestedOutputs) :
                parser->CreateNetworkFromTextFile(modelPath.c_str(), inputShapes, requestedOutputs));
        }

        for (const std::string& inputLayerName : params.m_InputBindings)
        {
            inputBindings.push_back(parser->GetNetworkInputBindingInfo(inputLayerName));
        }

        for (const std::string& outputLayerName : params.m_OutputBindings)
        {
            outputBindings.push_back(parser->GetNetworkOutputBindingInfo(outputLayerName));
        }

        return network;
    }
};

#if defined(ARMNN_SERIALIZER)
template <>
struct CreateNetworkImpl<armnnDeserializer::IDeserializer>
{
public:
    using IParser          = armnnDeserializer::IDeserializer;
    using Params           = InferenceModelInternal::Params;

    static armnn::INetworkPtr Create(const Params& params,
                                     std::vector<armnn::BindingPointInfo>& inputBindings,
                                     std::vector<armnn::BindingPointInfo>& outputBindings)
    {
        auto parser(IParser::Create());
        BOOST_ASSERT(parser);

        armnn::INetworkPtr network{nullptr, [](armnn::INetwork *){}};

        {
            ARMNN_SCOPED_HEAP_PROFILING("Parsing");

            boost::system::error_code errorCode;
            boost::filesystem::path pathToFile(params.m_ModelPath);
            if (!boost::filesystem::exists(pathToFile, errorCode))
            {
                throw armnn::FileNotFoundException(boost::str(
                                                   boost::format("Cannot find the file (%1%) errorCode: %2% %3%") %
                                                   params.m_ModelPath %
                                                   errorCode %
                                                   CHECK_LOCATION().AsString()));
            }
            std::ifstream file(params.m_ModelPath, std::ios::binary);

            network = parser->CreateNetworkFromBinary(file);
        }

        unsigned int subgraphId = boost::numeric_cast<unsigned int>(params.m_SubgraphId);

        for (const std::string& inputLayerName : params.m_InputBindings)
        {
            armnnDeserializer::BindingPointInfo inputBinding =
                parser->GetNetworkInputBindingInfo(subgraphId, inputLayerName);
            inputBindings.push_back(std::make_pair(inputBinding.m_BindingId, inputBinding.m_TensorInfo));
        }

        for (const std::string& outputLayerName : params.m_OutputBindings)
        {
            armnnDeserializer::BindingPointInfo outputBinding =
                parser->GetNetworkOutputBindingInfo(subgraphId, outputLayerName);
            outputBindings.push_back(std::make_pair(outputBinding.m_BindingId, outputBinding.m_TensorInfo));
        }

        return network;
    }
};
#endif

#if defined(ARMNN_TF_LITE_PARSER)
template <>
struct CreateNetworkImpl<armnnTfLiteParser::ITfLiteParser>
{
public:
    using IParser = armnnTfLiteParser::ITfLiteParser;
    using Params = InferenceModelInternal::Params;

    static armnn::INetworkPtr Create(const Params& params,
                                     std::vector<armnn::BindingPointInfo>& inputBindings,
                                     std::vector<armnn::BindingPointInfo>& outputBindings)
    {
        const std::string& modelPath = params.m_ModelPath;

        // Create a network from a file on disk
        auto parser(IParser::Create());

        armnn::INetworkPtr network{nullptr, [](armnn::INetwork *){}};

        {
            ARMNN_SCOPED_HEAP_PROFILING("Parsing");
            network = parser->CreateNetworkFromBinaryFile(modelPath.c_str());
        }

        for (const std::string& inputLayerName : params.m_InputBindings)
        {
            armnn::BindingPointInfo inputBinding =
                parser->GetNetworkInputBindingInfo(params.m_SubgraphId, inputLayerName);
            inputBindings.push_back(inputBinding);
        }

        for (const std::string& outputLayerName : params.m_OutputBindings)
        {
            armnn::BindingPointInfo outputBinding =
                parser->GetNetworkOutputBindingInfo(params.m_SubgraphId, outputLayerName);
            outputBindings.push_back(outputBinding);
        }

        return network;
    }
};
#endif

#if defined(ARMNN_ONNX_PARSER)
template <>
struct CreateNetworkImpl<armnnOnnxParser::IOnnxParser>
{
public:
    using IParser = armnnOnnxParser::IOnnxParser;
    using Params = InferenceModelInternal::Params;
    using BindingPointInfo = InferenceModelInternal::BindingPointInfo;

    static armnn::INetworkPtr Create(const Params& params,
                                     std::vector<BindingPointInfo>& inputBindings,
                                     std::vector<BindingPointInfo>& outputBindings)
    {
        const std::string& modelPath = params.m_ModelPath;

        // Create a network from a file on disk
        auto parser(IParser::Create());

        armnn::INetworkPtr network{nullptr, [](armnn::INetwork *){}};

        {
            ARMNN_SCOPED_HEAP_PROFILING("Parsing");
            network = (params.m_IsModelBinary ?
                parser->CreateNetworkFromBinaryFile(modelPath.c_str()) :
                parser->CreateNetworkFromTextFile(modelPath.c_str()));
        }

        for (const std::string& inputLayerName : params.m_InputBindings)
        {
            BindingPointInfo inputBinding = parser->GetNetworkInputBindingInfo(inputLayerName);
            inputBindings.push_back(inputBinding);
        }

        for (const std::string& outputLayerName : params.m_OutputBindings)
        {
            BindingPointInfo outputBinding = parser->GetNetworkOutputBindingInfo(outputLayerName);
            outputBindings.push_back(outputBinding);
        }

        return network;
    }
};
#endif



template <typename IParser, typename TDataType>
class InferenceModel
{
public:
    using DataType           = TDataType;
    using Params             = InferenceModelInternal::Params;
    using QuantizationParams = InferenceModelInternal::QuantizationParams;
    using TContainer         = boost::variant<std::vector<float>, std::vector<int>, std::vector<unsigned char>>;

    struct CommandLineOptions
    {
        std::string m_ModelDir;
        std::vector<std::string> m_ComputeDevices;
        std::string m_DynamicBackendsPath;
        bool m_VisualizePostOptimizationModel;
        bool m_EnableFp16TurboMode;
        std::string m_Labels;

        std::vector<armnn::BackendId> GetComputeDevicesAsBackendIds()
        {
            std::vector<armnn::BackendId> backendIds;
            std::copy(m_ComputeDevices.begin(), m_ComputeDevices.end(), std::back_inserter(backendIds));
            return backendIds;
        }
    };

    static void AddCommandLineOptions(boost::program_options::options_description& desc, CommandLineOptions& options)
    {
        namespace po = boost::program_options;

        const std::vector<std::string> defaultComputes = { "CpuAcc", "CpuRef" };

        const std::string backendsMessage = "Which device to run layers on by default. Possible choices: "
                                          + armnn::BackendRegistryInstance().GetBackendIdsAsString();

        desc.add_options()
            ("model-dir,m", po::value<std::string>(&options.m_ModelDir)->required(),
                "Path to directory containing model files (.caffemodel/.prototxt/.tflite)")
            ("compute,c", po::value<std::vector<std::string>>(&options.m_ComputeDevices)->
                default_value(defaultComputes, boost::algorithm::join(defaultComputes, ", "))->
                multitoken(), backendsMessage.c_str())
            ("dynamic-backends-path,b", po::value(&options.m_DynamicBackendsPath),
                "Path where to load any available dynamic backend from. "
                "If left empty (the default), dynamic backends will not be used.")
            ("labels,l", po::value<std::string>(&options.m_Labels),
                "Text file containing one image filename - correct label pair per line, "
                "used to test the accuracy of the network.")
            ("visualize-optimized-model,v",
                po::value<bool>(&options.m_VisualizePostOptimizationModel)->default_value(false),
             "Produce a dot file useful for visualizing the graph post optimization."
                "The file will have the same name as the model with the .dot extention.")
            ("fp16-turbo-mode", po::value<bool>(&options.m_EnableFp16TurboMode)->default_value(false),
                "If this option is enabled FP32 layers, weights and biases will be converted "
                "to FP16 where the backend supports it.");
    }

    InferenceModel(const Params& params,
                   bool enableProfiling,
                   const std::string& dynamicBackendsPath,
                   const std::shared_ptr<armnn::IRuntime>& runtime = nullptr)
        : m_EnableProfiling(enableProfiling)
        , m_DynamicBackendsPath(dynamicBackendsPath)
    {
        if (runtime)
        {
            m_Runtime = runtime;
        }
        else
        {
            armnn::IRuntime::CreationOptions options;
            options.m_EnableGpuProfiling = m_EnableProfiling;
            options.m_DynamicBackendsPath = m_DynamicBackendsPath;
            m_Runtime = std::move(armnn::IRuntime::Create(options));
        }

        std::string invalidBackends;
        if (!CheckRequestedBackendsAreValid(params.m_ComputeDevices, armnn::Optional<std::string&>(invalidBackends)))
        {
            throw armnn::Exception("Some backend IDs are invalid: " + invalidBackends);
        }

        armnn::INetworkPtr network = CreateNetworkImpl<IParser>::Create(params, m_InputBindings, m_OutputBindings);

        armnn::IOptimizedNetworkPtr optNet{nullptr, [](armnn::IOptimizedNetwork*){}};
        {
            ARMNN_SCOPED_HEAP_PROFILING("Optimizing");

            armnn::OptimizerOptions options;
            options.m_ReduceFp32ToFp16 = params.m_EnableFp16TurboMode;

            optNet = armnn::Optimize(*network, params.m_ComputeDevices, m_Runtime->GetDeviceSpec(), options);
            if (!optNet)
            {
                throw armnn::Exception("Optimize returned nullptr");
            }
        }

        if (params.m_VisualizePostOptimizationModel)
        {
            boost::filesystem::path filename = params.m_ModelPath;
            filename.replace_extension("dot");
            std::fstream file(filename.c_str(), std::ios_base::out);
            optNet->SerializeToDot(file);
        }

        armnn::Status ret;
        {
            ARMNN_SCOPED_HEAP_PROFILING("LoadNetwork");
            ret = m_Runtime->LoadNetwork(m_NetworkIdentifier, std::move(optNet));
        }

        if (ret == armnn::Status::Failure)
        {
            throw armnn::Exception("IRuntime::LoadNetwork failed");
        }
    }

    void CheckInputIndexIsValid(unsigned int inputIndex) const
    {
        if (m_InputBindings.size() < inputIndex + 1)
        {
            throw armnn::Exception(boost::str(boost::format("Input index out of range: %1%") % inputIndex));
        }
    }

    void CheckOutputIndexIsValid(unsigned int outputIndex) const
    {
        if (m_OutputBindings.size() < outputIndex + 1)
        {
            throw armnn::Exception(boost::str(boost::format("Output index out of range: %1%") % outputIndex));
        }
    }

    unsigned int GetOutputSize(unsigned int outputIndex = 0u) const
    {
        CheckOutputIndexIsValid(outputIndex);
        return m_OutputBindings[outputIndex].second.GetNumElements();
    }

    std::chrono::duration<double, std::milli> Run(
            const std::vector<TContainer>& inputContainers,
            std::vector<TContainer>& outputContainers)
    {
        for (unsigned int i = 0; i < outputContainers.size(); ++i)
        {
            const unsigned int expectedOutputDataSize = GetOutputSize(i);

            boost::apply_visitor([expectedOutputDataSize, i](auto&& value)
            {
                const unsigned int actualOutputDataSize   = boost::numeric_cast<unsigned int>(value.size());
                if (actualOutputDataSize < expectedOutputDataSize)
                {
                    unsigned int outputIndex = boost::numeric_cast<unsigned int>(i);
                    throw armnn::Exception(
                            boost::str(boost::format("Not enough data for output #%1%: expected "
                            "%2% elements, got %3%") % outputIndex % expectedOutputDataSize % actualOutputDataSize));
                }
            },
            outputContainers[i]);
        }

        std::shared_ptr<armnn::IProfiler> profiler = m_Runtime->GetProfiler(m_NetworkIdentifier);
        if (profiler)
        {
            profiler->EnableProfiling(m_EnableProfiling);
        }

        // Start timer to record inference time in EnqueueWorkload (in milliseconds)
        const auto start_time = GetCurrentTime();

        armnn::Status ret = m_Runtime->EnqueueWorkload(m_NetworkIdentifier,
                                                       MakeInputTensors(inputContainers),
                                                       MakeOutputTensors(outputContainers));

        const auto end_time = GetCurrentTime();

        // if profiling is enabled print out the results
        if (profiler && profiler->IsProfilingEnabled())
        {
            profiler->Print(std::cout);
        }

        if (ret == armnn::Status::Failure)
        {
            throw armnn::Exception("IRuntime::EnqueueWorkload failed");
        }
        else
        {
            return std::chrono::duration<double, std::milli>(end_time - start_time);
        }
    }

    const armnn::BindingPointInfo& GetInputBindingInfo(unsigned int inputIndex = 0u) const
    {
        CheckInputIndexIsValid(inputIndex);
        return m_InputBindings[inputIndex];
    }

    const std::vector<armnn::BindingPointInfo>& GetInputBindingInfos() const
    {
        return m_InputBindings;
    }

    const armnn::BindingPointInfo& GetOutputBindingInfo(unsigned int outputIndex = 0u) const
    {
        CheckOutputIndexIsValid(outputIndex);
        return m_OutputBindings[outputIndex];
    }

    const std::vector<armnn::BindingPointInfo>& GetOutputBindingInfos() const
    {
        return m_OutputBindings;
    }

    QuantizationParams GetQuantizationParams(unsigned int outputIndex = 0u) const
    {
        CheckOutputIndexIsValid(outputIndex);
        return std::make_pair(m_OutputBindings[outputIndex].second.GetQuantizationScale(),
                              m_OutputBindings[outputIndex].second.GetQuantizationOffset());
    }

    QuantizationParams GetInputQuantizationParams(unsigned int inputIndex = 0u) const
    {
        CheckInputIndexIsValid(inputIndex);
        return std::make_pair(m_InputBindings[inputIndex].second.GetQuantizationScale(),
                              m_InputBindings[inputIndex].second.GetQuantizationOffset());
    }

    std::vector<QuantizationParams> GetAllQuantizationParams() const
    {
        std::vector<QuantizationParams> quantizationParams;
        for (unsigned int i = 0u; i < m_OutputBindings.size(); i++)
        {
            quantizationParams.push_back(GetQuantizationParams(i));
        }
        return quantizationParams;
    }

private:
    armnn::NetworkId m_NetworkIdentifier;
    std::shared_ptr<armnn::IRuntime> m_Runtime;

    std::vector<armnn::BindingPointInfo> m_InputBindings;
    std::vector<armnn::BindingPointInfo> m_OutputBindings;
    bool m_EnableProfiling;
    std::string m_DynamicBackendsPath;

    template<typename TContainer>
    armnn::InputTensors MakeInputTensors(const std::vector<TContainer>& inputDataContainers)
    {
        return armnnUtils::MakeInputTensors(m_InputBindings, inputDataContainers);
    }

    template<typename TContainer>
    armnn::OutputTensors MakeOutputTensors(std::vector<TContainer>& outputDataContainers)
    {
        return armnnUtils::MakeOutputTensors(m_OutputBindings, outputDataContainers);
    }

    std::chrono::high_resolution_clock::time_point GetCurrentTime()
    {
        return std::chrono::high_resolution_clock::now();
    }

    std::chrono::duration<double, std::milli> GetTimeDuration(
            std::chrono::high_resolution_clock::time_point& start_time,
            std::chrono::high_resolution_clock::time_point& end_time)
    {
        return std::chrono::duration<double, std::milli>(end_time - start_time);
    }

};
