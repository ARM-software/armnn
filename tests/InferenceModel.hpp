//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once


#include <armnn/ArmNN.hpp>

#if !defined(ARMNN_DISABLE_THREADS)
#include <armnn/Threadpool.hpp>
#include <common/include/IgnoreUnused.hpp>
#endif

#include <armnn/Logging.hpp>
#include <armnn/utility/Timer.hpp>
#include <armnn/BackendRegistry.hpp>
#include <armnn/utility/Assert.hpp>
#include <armnn/utility/NumericCast.hpp>

#include <armnnUtils/TContainer.hpp>
#include "NetworkExecutionUtils/NetworkExecutionUtils.hpp"

#include <common/include/ProfilingGuid.hpp>

#if defined(ARMNN_SERIALIZER)
#include "armnnDeserializer/IDeserializer.hpp"
#endif
#if defined(ARMNN_TF_LITE_PARSER)
#include <armnnTfLiteParser/ITfLiteParser.hpp>
#endif
#if defined(ARMNN_ONNX_PARSER)
#include <armnnOnnxParser/IOnnxParser.hpp>
#endif

#include <armnnUtils/Filesystem.hpp>
#include <HeapProfiling.hpp>
#include <TensorIOUtils.hpp>

#include "armnn/utility/StringUtils.hpp"
#include <cxxopts/cxxopts.hpp>
#include "CxxoptsUtils.hpp"
#include <fmt/format.h>
#include <mapbox/variant.hpp>

#include <algorithm>
#include <iterator>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <type_traits>

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
    bool                            m_AllowExpandedDims;
    bool                            m_IsModelBinary;
    bool                            m_VisualizePostOptimizationModel;
    bool                            m_EnableFp16TurboMode;
    bool                            m_EnableBf16TurboMode;
    bool                            m_PrintIntermediateLayers;
    bool                            m_PrintIntermediateLayersToFile;
    bool                            m_ParseUnsupported;
    bool                            m_InferOutputShape;
    bool                            m_EnableFastMath;
    bool                            m_SaveCachedNetwork;
    bool                            m_OutputDetailsToStdOut;
    bool                            m_OutputDetailsOnlyToStdOut;
    std::string                     m_CachedNetworkFilePath;
    unsigned int                    m_NumberOfThreads;
    std::string                     m_MLGOTuningFilePath;
    bool                            m_AsyncEnabled;
    size_t                          m_ThreadPoolSize;
    bool                            m_ImportInputsIfAligned;


    Params()
        : m_ComputeDevices{}
        , m_SubgraphId(0)
        , m_AllowExpandedDims(false)
        , m_IsModelBinary(true)
        , m_VisualizePostOptimizationModel(false)
        , m_EnableFp16TurboMode(false)
        , m_EnableBf16TurboMode(false)
        , m_PrintIntermediateLayers(false)
        , m_PrintIntermediateLayersToFile(false)
        , m_ParseUnsupported(false)
        , m_InferOutputShape(false)
        , m_EnableFastMath(false)
        , m_SaveCachedNetwork(false)
        , m_OutputDetailsToStdOut(false)
        , m_OutputDetailsOnlyToStdOut(false)
        , m_CachedNetworkFilePath("")
        , m_NumberOfThreads(0)
        , m_MLGOTuningFilePath("")
        , m_AsyncEnabled(false)
        , m_ThreadPoolSize(0)
        , m_ImportInputsIfAligned(false)
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
                throw armnn::Exception(fmt::format(
                    "Not every input has its tensor shape specified: expected={0}, got={1}",
                    numInputBindings, numInputShapes));
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
        ARMNN_ASSERT(parser);

        armnn::INetworkPtr network{nullptr, [](armnn::INetwork *){}};

        {
            ARMNN_SCOPED_HEAP_PROFILING("Parsing");

            std::error_code errorCode;
            fs::path pathToFile(params.m_ModelPath);
            if (!fs::exists(pathToFile, errorCode))
            {
                throw armnn::FileNotFoundException(fmt::format("Cannot find the file ({0}) errorCode: {1} {2}",
                                                   params.m_ModelPath,
                                                   errorCode.message(),
                                                   CHECK_LOCATION().AsString()));
            }
            std::ifstream file(params.m_ModelPath, std::ios::binary);

            network = parser->CreateNetworkFromBinary(file);
        }

        unsigned int subgraphId = armnn::numeric_cast<unsigned int>(params.m_SubgraphId);

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
        IParser::TfLiteParserOptions options;
        options.m_AllowExpandedDims          = params.m_AllowExpandedDims;
        options.m_StandInLayerForUnsupported = params.m_ParseUnsupported;
        options.m_InferAndValidate           = params.m_InferOutputShape;
        auto parser(IParser::Create(options));

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

        std::map<std::string, armnn::TensorShape> inputShapes;
        if (!params.m_InputShapes.empty())
        {
            const size_t numInputShapes   = params.m_InputShapes.size();
            const size_t numInputBindings = params.m_InputBindings.size();
            if (numInputShapes < numInputBindings)
            {
                throw armnn::Exception(fmt::format(
                    "Not every input has its tensor shape specified: expected={0}, got={1}",
                    numInputBindings, numInputShapes));
            }

            for (size_t i = 0; i < numInputShapes; i++)
            {
                inputShapes[params.m_InputBindings[i]] = params.m_InputShapes[i];
            }

            {
                ARMNN_SCOPED_HEAP_PROFILING("Parsing");
                network = (params.m_IsModelBinary ?
                    parser->CreateNetworkFromBinaryFile(modelPath.c_str(), inputShapes) :
                    parser->CreateNetworkFromTextFile(modelPath.c_str(), inputShapes));
            }
        }

        else
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


    struct CommandLineOptions
    {
        std::string m_ModelDir;
        std::vector<std::string> m_ComputeDevices;
        std::string m_DynamicBackendsPath;
        bool m_VisualizePostOptimizationModel;
        bool m_EnableFp16TurboMode;
        bool m_EnableBf16TurboMode;
        std::string m_Labels;

        std::vector<armnn::BackendId> GetComputeDevicesAsBackendIds()
        {
            std::vector<armnn::BackendId> backendIds;
            std::copy(m_ComputeDevices.begin(), m_ComputeDevices.end(), std::back_inserter(backendIds));
            return backendIds;
        }
    };

    static void AddCommandLineOptions(cxxopts::Options& options,
                                      CommandLineOptions& cLineOptions, std::vector<std::string>& required)
    {
        const std::vector<std::string> defaultComputes = { "CpuAcc", "CpuRef" };

        const std::string backendsMessage = "Which device to run layers on by default. Possible choices: "
                                          + armnn::BackendRegistryInstance().GetBackendIdsAsString();

        options
            .allow_unrecognised_options()
            .add_options()
                ("m,model-dir", "Path to directory containing model files (.prototxt/.tflite)",
                 cxxopts::value<std::string>(cLineOptions.m_ModelDir))
                ("c,compute", backendsMessage.c_str(),
                 cxxopts::value<std::vector<std::string>>(cLineOptions.m_ComputeDevices)->default_value("CpuRef"))
                ("b,dynamic-backends-path",
                 "Path where to load any available dynamic backend from. "
                 "If left empty (the default), dynamic backends will not be used.",
                 cxxopts::value(cLineOptions.m_DynamicBackendsPath))
                ("l,labels",
                 "Text file containing one image filename - correct label pair per line, "
                 "used to test the accuracy of the network.", cxxopts::value<std::string>(cLineOptions.m_Labels))
                ("v,visualize-optimized-model",
                 "Produce a dot file useful for visualizing the graph post optimization."
                 "The file will have the same name as the model with the .dot extention.",
                 cxxopts::value<bool>(cLineOptions.m_VisualizePostOptimizationModel)->default_value("false"))
                ("fp16-turbo-mode",
                 "If this option is enabled FP32 layers, weights and biases will be converted "
                 "to FP16 where the backend supports it.",
                 cxxopts::value<bool>(cLineOptions.m_EnableFp16TurboMode)->default_value("false"))
                ("bf16-turbo-mode",
                 "If this option is enabled FP32 layers, weights and biases will be converted "
                 "to BF16 where the backend supports it.",
                 cxxopts::value<bool>(cLineOptions.m_EnableBf16TurboMode)->default_value("false"));

        required.emplace_back("model-dir");
    }

    InferenceModel(const Params& params,
                   bool enableProfiling,
                   const std::string& dynamicBackendsPath,
                   const std::shared_ptr<armnn::IRuntime>& runtime = nullptr)
        : m_EnableProfiling(enableProfiling),
          m_ProfilingDetailsMethod(armnn::ProfilingDetailsMethod::Undefined),
          m_DynamicBackendsPath(dynamicBackendsPath),
          m_ImportInputsIfAligned(params.m_ImportInputsIfAligned)
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
            m_Runtime = armnn::IRuntime::Create(options);
        }

        // Configure the Profiler if the the profiling details are opted for
        if (params.m_OutputDetailsOnlyToStdOut)
            m_ProfilingDetailsMethod = armnn::ProfilingDetailsMethod::DetailsOnly;
        else if (params.m_OutputDetailsToStdOut)
            m_ProfilingDetailsMethod = armnn::ProfilingDetailsMethod::DetailsWithEvents;

        std::string invalidBackends;
        if (!CheckRequestedBackendsAreValid(params.m_ComputeDevices, armnn::Optional<std::string&>(invalidBackends)))
        {
            throw armnn::Exception("Some backend IDs are invalid: " + invalidBackends);
        }

        armnn::IOptimizedNetworkPtr optNet{nullptr, [](armnn::IOptimizedNetwork*){}};
        {
            const auto parsing_start_time = armnn::GetTimeNow();
            armnn::INetworkPtr network = CreateNetworkImpl<IParser>::Create(params, m_InputBindings, m_OutputBindings);

            ARMNN_LOG(info) << "Network parsing time: " << std::setprecision(2)
                            << std::fixed << armnn::GetTimeDuration(parsing_start_time).count() << " ms.";

            ARMNN_SCOPED_HEAP_PROFILING("Optimizing");

            armnn::OptimizerOptionsOpaque options;
            options.SetReduceFp32ToFp16(params.m_EnableFp16TurboMode);
            options.SetDebugEnabled(params.m_PrintIntermediateLayers);
            options.SetDebugToFileEnabled(params.m_PrintIntermediateLayersToFile);
            options.SetShapeInferenceMethod(params.m_InferOutputShape ?
                    armnn::ShapeInferenceMethod::InferAndValidate : armnn::ShapeInferenceMethod::ValidateOnly);
            options.SetProfilingEnabled(m_EnableProfiling);

            armnn::BackendOptions gpuAcc("GpuAcc",
            {
                { "FastMathEnabled", params.m_EnableFastMath },
                { "SaveCachedNetwork", params.m_SaveCachedNetwork },
                { "CachedNetworkFilePath", params.m_CachedNetworkFilePath },
                { "MLGOTuningFilePath", params.m_MLGOTuningFilePath }
            });

            armnn::BackendOptions cpuAcc("CpuAcc",
            {
                { "FastMathEnabled", params.m_EnableFastMath },
                { "NumberOfThreads", params.m_NumberOfThreads }
            });
            options.AddModelOption(gpuAcc);
            options.AddModelOption(cpuAcc);

            const auto optimization_start_time = armnn::GetTimeNow();
            optNet = armnn::Optimize(*network, params.m_ComputeDevices, m_Runtime->GetDeviceSpec(), options);

            ARMNN_LOG(info) << "Optimization time: " << std::setprecision(2)
                            << std::fixed << armnn::GetTimeDuration(optimization_start_time).count() << " ms.";

            if (!optNet)
            {
                throw armnn::Exception("Optimize returned nullptr");
            }


        }

        if (params.m_VisualizePostOptimizationModel)
        {
            fs::path filename = params.m_ModelPath;
            filename.replace_extension("dot");
            std::fstream file(filename.c_str(), std::ios_base::out);
            optNet->SerializeToDot(file);
        }

        armnn::Status ret;
        {
            ARMNN_SCOPED_HEAP_PROFILING("LoadNetwork");

            const auto loading_start_time = armnn::GetTimeNow();
            armnn::INetworkProperties networkProperties(params.m_AsyncEnabled,
                                                        armnn::MemorySource::Undefined,
                                                        armnn::MemorySource::Undefined,
                                                        enableProfiling,
                                                        m_ProfilingDetailsMethod);
            std::string errorMessage;
            ret = m_Runtime->LoadNetwork(m_NetworkIdentifier, std::move(optNet), errorMessage, networkProperties);

            ARMNN_LOG(info) << "Network loading time: " << std::setprecision(2)
                            << std::fixed << armnn::GetTimeDuration(loading_start_time).count() << " ms.";
#if !defined(ARMNN_DISABLE_THREADS)
            if (params.m_AsyncEnabled && params.m_ThreadPoolSize > 0)
            {
                std::vector<std::shared_ptr<armnn::IWorkingMemHandle>> memHandles;
                for (size_t i = 0; i < params.m_ThreadPoolSize; ++i)
                {
                    memHandles.emplace_back(m_Runtime->CreateWorkingMemHandle(m_NetworkIdentifier));
                }

                m_Threadpool = std::make_unique<armnn::Threadpool>(params.m_ThreadPoolSize,
                                                                   m_Runtime.get(),
                                                                   memHandles);
            }
#endif
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
            throw armnn::Exception(fmt::format("Input index out of range: {}", inputIndex));
        }
    }

    void CheckOutputIndexIsValid(unsigned int outputIndex) const
    {
        if (m_OutputBindings.size() < outputIndex + 1)
        {
            throw armnn::Exception(fmt::format("Output index out of range: {}", outputIndex));
        }
    }

    unsigned int GetInputSize(unsigned int inputIndex = 0u) const
    {
        CheckInputIndexIsValid(inputIndex);
        return m_InputBindings[inputIndex].second.GetNumElements();
    }

    unsigned int GetOutputSize(unsigned int outputIndex = 0u) const
    {
        CheckOutputIndexIsValid(outputIndex);
        return m_OutputBindings[outputIndex].second.GetNumElements();
    }

    std::chrono::duration<double, std::milli> Run(
            const std::vector<armnnUtils::TContainer>& inputContainers,
            std::vector<armnnUtils::TContainer>& outputContainers)
    {
        for (unsigned int i = 0; i < outputContainers.size(); ++i)
        {
            const unsigned int expectedOutputDataSize = GetOutputSize(i);

            mapbox::util::apply_visitor([expectedOutputDataSize, i](auto&& value)
            {
                const unsigned int actualOutputDataSize   = armnn::numeric_cast<unsigned int>(value.size());
                if (actualOutputDataSize < expectedOutputDataSize)
                {
                    unsigned int outputIndex = i;
                    throw armnn::Exception(
                            fmt::format("Not enough data for output #{0}: expected "
                            "{1} elements, got {2}", outputIndex, expectedOutputDataSize, actualOutputDataSize));
                }
            },
            outputContainers[i]);
        }

        std::shared_ptr<armnn::IProfiler> profiler = m_Runtime->GetProfiler(m_NetworkIdentifier);

        // Start timer to record inference time in EnqueueWorkload (in milliseconds)
        const auto start_time = armnn::GetTimeNow();

        armnn::Status ret;
        if (m_ImportInputsIfAligned)
        {
            std::vector<armnn::ImportedInputId> importedInputIds = m_Runtime->ImportInputs(
                m_NetworkIdentifier, MakeInputTensors(inputContainers), armnn::MemorySource::Malloc);

            std::vector<armnn::ImportedOutputId> importedOutputIds = m_Runtime->ImportOutputs(
                m_NetworkIdentifier, MakeOutputTensors(outputContainers), armnn::MemorySource::Malloc);

            ret = m_Runtime->EnqueueWorkload(m_NetworkIdentifier,
                                             MakeInputTensors(inputContainers),
                                             MakeOutputTensors(outputContainers),
                                             importedInputIds,
                                             importedOutputIds);
        }
        else
        {
            ret = m_Runtime->EnqueueWorkload(m_NetworkIdentifier,
                                             MakeInputTensors(inputContainers),
                                             MakeOutputTensors(outputContainers));
        }
        const auto duration = armnn::GetTimeDuration(start_time);

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
            return duration;
        }
    }

    std::tuple<unsigned int, std::chrono::duration<double, std::milli>> RunAsync(
        armnn::experimental::IWorkingMemHandle& workingMemHandleRef,
        const std::vector<armnnUtils::TContainer>& inputContainers,
        std::vector<armnnUtils::TContainer>& outputContainers,
        unsigned int inferenceID)
    {
        for (unsigned int i = 0; i < outputContainers.size(); ++i)
        {
            const unsigned int expectedOutputDataSize = GetOutputSize(i);

            mapbox::util::apply_visitor([expectedOutputDataSize, i](auto&& value)
            {
                const unsigned int actualOutputDataSize   = armnn::numeric_cast<unsigned int>(value.size());
                if (actualOutputDataSize < expectedOutputDataSize)
                {
                    unsigned int outputIndex = i;
                    throw armnn::Exception(
                            fmt::format("Not enough data for output #{0}: expected "
                            "{1} elements, got {2}", outputIndex, expectedOutputDataSize, actualOutputDataSize));
                }
            },
            outputContainers[i]);
        }

        std::shared_ptr<armnn::IProfiler> profiler = m_Runtime->GetProfiler(m_NetworkIdentifier);

        // Start timer to record inference time in EnqueueWorkload (in milliseconds)
        const auto start_time = armnn::GetTimeNow();

        armnn::Status ret = m_Runtime->Execute(workingMemHandleRef,
                                               MakeInputTensors(inputContainers),
                                               MakeOutputTensors(outputContainers));

        const auto duration = armnn::GetTimeDuration(start_time);

        // if profiling is enabled print out the results
        if (profiler && profiler->IsProfilingEnabled())
        {
            profiler->Print(std::cout);
        }

        if (ret == armnn::Status::Failure)
        {
            throw armnn::Exception(
                fmt::format("IRuntime::Execute asynchronously failed for network #{0} on inference #{1}",
                            m_NetworkIdentifier, inferenceID));
        }
        else
        {
            return std::make_tuple(inferenceID, duration);
        }
    }

    void RunAsync(const std::vector<armnnUtils::TContainer>& inputContainers,
                  std::vector<armnnUtils::TContainer>& outputContainers,
                  std::shared_ptr<armnn::IAsyncExecutionCallback> cb)
    {
#if !defined(ARMNN_DISABLE_THREADS)
        for (unsigned int i = 0; i < outputContainers.size(); ++i)
        {
            const unsigned int expectedOutputDataSize = GetOutputSize(i);

            mapbox::util::apply_visitor([expectedOutputDataSize, i](auto&& value)
            {
                const unsigned int actualOutputDataSize   = armnn::numeric_cast<unsigned int>(value.size());
                if (actualOutputDataSize < expectedOutputDataSize)
                {
                    unsigned int outputIndex = i;
                    throw armnn::Exception(
                            fmt::format("Not enough data for output #{0}: expected "
                            "{1} elements, got {2}", outputIndex, expectedOutputDataSize, actualOutputDataSize));
                }
            },
            outputContainers[i]);
        }

        std::shared_ptr<armnn::IProfiler> profiler = m_Runtime->GetProfiler(m_NetworkIdentifier);

        m_Threadpool->Schedule(m_NetworkIdentifier,
                               MakeInputTensors(inputContainers),
                               MakeOutputTensors(outputContainers),
                               armnn::QosExecPriority::Medium,
                               cb);

        // if profiling is enabled print out the results
        if (profiler && profiler->IsProfilingEnabled())
        {
            profiler->Print(std::cout);
        }
#endif
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

    std::unique_ptr<armnn::experimental::IWorkingMemHandle> CreateWorkingMemHandle()
    {
        return m_Runtime->CreateWorkingMemHandle(m_NetworkIdentifier);
    }

private:
    armnn::NetworkId m_NetworkIdentifier;
    std::shared_ptr<armnn::IRuntime> m_Runtime;
#if !defined(ARMNN_DISABLE_THREADS)
    std::unique_ptr<armnn::Threadpool> m_Threadpool;
#endif

    std::vector<armnn::BindingPointInfo> m_InputBindings;
    std::vector<armnn::BindingPointInfo> m_OutputBindings;
    bool m_EnableProfiling;
    armnn::ProfilingDetailsMethod m_ProfilingDetailsMethod;
    std::string m_DynamicBackendsPath;
    bool m_ImportInputsIfAligned;

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
};
