//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once
#include "armnn/ArmNN.hpp"

#if defined(ARMNN_TF_LITE_PARSER)
#include "armnnTfLiteParser/ITfLiteParser.hpp"
#endif

#include <HeapProfiling.hpp>
#if defined(ARMNN_ONNX_PARSER)
#include "armnnOnnxParser/IOnnxParser.hpp"
#endif

#include <boost/exception/exception.hpp>
#include <boost/exception/diagnostic_information.hpp>
#include <boost/log/trivial.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include <map>
#include <string>
#include <fstream>
#include <type_traits>

namespace InferenceModelInternal
{
// This needs to go when the armnnCaffeParser, armnnTfParser and armnnTfLiteParser
// definitions of BindingPointInfo gets consolidated.
using BindingPointInfo = std::pair<armnn::LayerBindingId, armnn::TensorInfo>;

using QuantizationParams = std::pair<float,int32_t>;

struct Params
{
    std::string m_ModelPath;
    std::string m_InputBinding;
    std::string m_OutputBinding;
    const armnn::TensorShape* m_InputTensorShape;
    std::vector<armnn::Compute> m_ComputeDevice;
    bool m_EnableProfiling;
    size_t m_SubgraphId;
    bool m_IsModelBinary;
    bool m_VisualizePostOptimizationModel;
    bool m_EnableFp16TurboMode;

    Params()
        : m_InputTensorShape(nullptr)
        , m_ComputeDevice{armnn::Compute::CpuRef}
        , m_EnableProfiling(false)
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
    using BindingPointInfo = InferenceModelInternal::BindingPointInfo;

    static armnn::INetworkPtr Create(const Params& params,
                                     BindingPointInfo& inputBindings,
                                     BindingPointInfo& outputBindings)
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
      armnn::INetworkPtr network{nullptr, [](armnn::INetwork *){}};

      {
          ARMNN_SCOPED_HEAP_PROFILING("Parsing");
          // Handle text and binary input differently by calling the corresponding parser function
          network = (params.m_IsModelBinary ?
              parser->CreateNetworkFromBinaryFile(modelPath.c_str(), inputShapes, requestedOutputs) :
              parser->CreateNetworkFromTextFile(modelPath.c_str(), inputShapes, requestedOutputs));
      }

      inputBindings  = parser->GetNetworkInputBindingInfo(params.m_InputBinding);
      outputBindings = parser->GetNetworkOutputBindingInfo(params.m_OutputBinding);
      return network;
    }
};

#if defined(ARMNN_TF_LITE_PARSER)
template <>
struct CreateNetworkImpl<armnnTfLiteParser::ITfLiteParser>
{
public:
    using IParser = armnnTfLiteParser::ITfLiteParser;
    using Params = InferenceModelInternal::Params;
    using BindingPointInfo = InferenceModelInternal::BindingPointInfo;

    static armnn::INetworkPtr Create(const Params& params,
                                     BindingPointInfo& inputBindings,
                                     BindingPointInfo& outputBindings)
    {
      const std::string& modelPath = params.m_ModelPath;

      // Create a network from a file on disk
      auto parser(IParser::Create());

      armnn::INetworkPtr network{nullptr, [](armnn::INetwork *){}};

      {
          ARMNN_SCOPED_HEAP_PROFILING("Parsing");
          network = parser->CreateNetworkFromBinaryFile(modelPath.c_str());
      }

      inputBindings  = parser->GetNetworkInputBindingInfo(params.m_SubgraphId, params.m_InputBinding);
      outputBindings = parser->GetNetworkOutputBindingInfo(params.m_SubgraphId, params.m_OutputBinding);
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
                                     BindingPointInfo& inputBindings,
                                     BindingPointInfo& outputBindings)
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

      inputBindings  = parser->GetNetworkInputBindingInfo(params.m_InputBinding);
      outputBindings = parser->GetNetworkOutputBindingInfo(params.m_OutputBinding);
      return network;
    }
};
#endif

template<typename TContainer>
inline armnn::InputTensors MakeInputTensors(const InferenceModelInternal::BindingPointInfo& input,
    const TContainer& inputTensorData)
{
    if (inputTensorData.size() != input.second.GetNumElements())
    {
        try
        {
            throw armnn::Exception(boost::str(boost::format("Input tensor has incorrect size. Expected %1% elements "
                "but got %2%.") % input.second.GetNumElements() % inputTensorData.size()));
        } catch (const boost::exception& e)
        {
            // Coverity fix: it should not be possible to get here but boost::str and boost::format can both
            // throw uncaught exceptions, convert them to armnn exceptions and rethrow.
            throw armnn::Exception(diagnostic_information(e));
        }
    }
    return { { input.first, armnn::ConstTensor(input.second, inputTensorData.data()) } };
}

template<typename TContainer>
inline armnn::OutputTensors MakeOutputTensors(const InferenceModelInternal::BindingPointInfo& output,
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
    using Params = InferenceModelInternal::Params;

    struct CommandLineOptions
    {
        std::string m_ModelDir;
        std::vector<armnn::Compute> m_ComputeDevice;
        bool m_VisualizePostOptimizationModel;
        bool m_EnableFp16TurboMode;
    };

    static void AddCommandLineOptions(boost::program_options::options_description& desc, CommandLineOptions& options)
    {
        namespace po = boost::program_options;

        desc.add_options()
            ("model-dir,m", po::value<std::string>(&options.m_ModelDir)->required(),
                "Path to directory containing model files (.caffemodel/.prototxt/.tflite)")
            ("compute,c", po::value<std::vector<armnn::Compute>>(&options.m_ComputeDevice)->default_value
                 ({armnn::Compute::CpuAcc, armnn::Compute::CpuRef}),
                "Which device to run layers on by default. Possible choices: CpuAcc, CpuRef, GpuAcc")
            ("visualize-optimized-model,v",
                po::value<bool>(&options.m_VisualizePostOptimizationModel)->default_value(false),
             "Produce a dot file useful for visualizing the graph post optimization."
                "The file will have the same name as the model with the .dot extention.")
            ("fp16-turbo-mode", po::value<bool>(&options.m_EnableFp16TurboMode)->default_value(false),
                "If this option is enabled FP32 layers, weights and biases will be converted "
                "to FP16 where the backend supports it.");
    }

    InferenceModel(const Params& params, const std::shared_ptr<armnn::IRuntime>& runtime = nullptr)
        : m_EnableProfiling(params.m_EnableProfiling)
    {
        if (runtime)
        {
            m_Runtime = runtime;
        }
        else
        {
            armnn::IRuntime::CreationOptions options;
            options.m_EnableGpuProfiling = m_EnableProfiling;
            m_Runtime = std::move(armnn::IRuntime::Create(options));
        }

        armnn::INetworkPtr network = CreateNetworkImpl<IParser>::Create(params, m_InputBindingInfo,
           m_OutputBindingInfo);

        armnn::IOptimizedNetworkPtr optNet{nullptr, [](armnn::IOptimizedNetwork *){}};
        {
            ARMNN_SCOPED_HEAP_PROFILING("Optimizing");

            armnn::OptimizerOptions options;
            options.m_ReduceFp32ToFp16 = params.m_EnableFp16TurboMode;

            optNet = armnn::Optimize(*network, params.m_ComputeDevice, m_Runtime->GetDeviceSpec(), options);
            if (!optNet)
            {
                throw armnn::Exception("Optimize returned nullptr");
            }
        }

        if (params.m_VisualizePostOptimizationModel)
        {
            boost::filesystem::path filename = params.m_ModelPath;
            filename.replace_extension("dot");
            std::fstream file(filename.c_str(),file.out);
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

    unsigned int GetOutputSize() const
    {
        return m_OutputBindingInfo.second.GetNumElements();
    }

    void Run(const std::vector<TDataType>& input, std::vector<TDataType>& output)
    {
        BOOST_ASSERT(output.size() == GetOutputSize());

        std::shared_ptr<armnn::IProfiler> profiler = m_Runtime->GetProfiler(m_NetworkIdentifier);
        if (profiler)
        {
            profiler->EnableProfiling(m_EnableProfiling);
        }

        armnn::Status ret = m_Runtime->EnqueueWorkload(m_NetworkIdentifier,
                                                       MakeInputTensors(input),
                                                       MakeOutputTensors(output));

        // if profiling is enabled print out the results
        if (profiler && profiler->IsProfilingEnabled())
        {
            profiler->Print(std::cout);
        }

        if (ret == armnn::Status::Failure)
        {
            throw armnn::Exception("IRuntime::EnqueueWorkload failed");
        }
    }

    const InferenceModelInternal::BindingPointInfo & GetInputBindingInfo() const
    {
        return m_InputBindingInfo;
    }

    const InferenceModelInternal::BindingPointInfo & GetOutputBindingInfo() const
    {
        return m_OutputBindingInfo;
    }

    InferenceModelInternal::QuantizationParams GetQuantizationParams() const
    {
        return std::make_pair(m_OutputBindingInfo.second.GetQuantizationScale(),
                              m_OutputBindingInfo.second.GetQuantizationOffset());
    }

private:
    armnn::NetworkId m_NetworkIdentifier;
    std::shared_ptr<armnn::IRuntime> m_Runtime;

    InferenceModelInternal::BindingPointInfo m_InputBindingInfo;
    InferenceModelInternal::BindingPointInfo m_OutputBindingInfo;
    bool m_EnableProfiling;

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
};
