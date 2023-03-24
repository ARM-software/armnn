//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//


#include "ArmNNExecutor.hpp"
#include "NetworkExecutionUtils/NetworkExecutionUtils.hpp"

#include <armnn/IAsyncExecutionCallback.hpp>
#include <AsyncExecutionCallback.hpp>


using namespace armnn;
using namespace std::chrono;

ArmNNExecutor::ArmNNExecutor(const ExecuteNetworkParams& params, armnn::IRuntime::CreationOptions runtimeOptions)
: m_Params(params)
{
    runtimeOptions.m_EnableGpuProfiling = params.m_EnableProfiling;
    runtimeOptions.m_DynamicBackendsPath = params.m_DynamicBackendsPath;

    // Create/Get the static ArmNN Runtime. Note that the m_Runtime will be shared by all ArmNNExecutor
    // instances so the RuntimeOptions cannot be altered for different ArmNNExecutor instances.
    m_Runtime = GetRuntime(runtimeOptions);

    auto parser = CreateParser();
    auto network = parser->CreateNetwork(m_Params);
    auto optNet = OptimizeNetwork(network.get());

    m_IOInfo = GetIOInfo(optNet.get());

    armnn::ProfilingDetailsMethod profilingDetailsMethod = ProfilingDetailsMethod::Undefined;
    if (params.m_OutputDetailsOnlyToStdOut)
    {
        profilingDetailsMethod = armnn::ProfilingDetailsMethod::DetailsOnly;
    }
    else if (params.m_OutputDetailsToStdOut)
    {
        profilingDetailsMethod = armnn::ProfilingDetailsMethod::DetailsWithEvents;
    }

    INetworkProperties networkProperties{m_Params.m_Concurrent,
                                         MemorySource::Undefined,
                                         MemorySource::Undefined,
                                         params.m_EnableProfiling,
                                         profilingDetailsMethod};

    std::string errorMsg;
    Status status = m_Runtime->LoadNetwork(m_NetworkId, std::move(optNet), errorMsg, networkProperties);
    if (status != Status::Success)
    {
        std::string message("Failed to create Arm NN Executor: ");
        message.append(errorMsg);
        // Throwing an exception at this point in the constructor causes lots of problems. We'll instead mark this
        // executor as not constructed.
        ARMNN_LOG(fatal) << message;
        m_constructionFailed = true;
        return;
    }

    SetupInputsAndOutputs();

    if (m_Params.m_Iterations > 1)
    {
        std::stringstream msg;
        msg << "Network will be executed " << m_Params.m_Iterations;
        if (m_Params.m_Concurrent)
        {
            msg << " times in an asynchronous manner. ";
        }
        else
        {
            msg << " times successively. ";
        }
        msg << "The input-tensor-data files will be reused recursively if the user didn't provide enough to "
               "cover each execution.";
        ARMNN_LOG(info) << msg.str();
    }

    if (m_Params.m_GenerateTensorData)
    {
        ARMNN_LOG(warning) << "The input data was generated, note that the output will not be useful";
    }

    if (m_Params.m_DontPrintOutputs)
    {
        ARMNN_LOG(info) << "Printing outputs to console is disabled.";
    }
}

void ArmNNExecutor::ExecuteAsync()
{
#if !defined(ARMNN_DISABLE_THREADS)
    std::vector<std::shared_ptr<armnn::IWorkingMemHandle>> memHandles;
    std::unique_ptr<armnn::Threadpool> threadpool;
    armnn::AsyncCallbackManager callbackManager;
    std::unordered_map<armnn::InferenceId, const armnn::OutputTensors*> inferenceOutputMap;

    for (size_t i = 0; i < m_Params.m_ThreadPoolSize; ++i)
    {
        memHandles.emplace_back(m_Runtime->CreateWorkingMemHandle(m_NetworkId));
    }

    threadpool = std::make_unique<armnn::Threadpool>(m_Params.m_ThreadPoolSize,
                                                     m_Runtime,
                                                     memHandles);

    ARMNN_LOG(info) << "Asynchronous Execution with Arm NN thread pool...  \n";
    // Declare the latest and earliest inference times here to be used when calculating overall time
    std::chrono::high_resolution_clock::time_point earliestStartTime =
            std::chrono::high_resolution_clock::time_point::max();
    std::chrono::high_resolution_clock::time_point latestEndTime =
            std::chrono::high_resolution_clock::now();

    // For the asynchronous execution, we are adding a pool of working memory handles (1 per thread) in the
    // LoadedNetwork with each scheduled inference having a specific priority
    for (size_t i = 0; i < m_Params.m_Iterations; ++i)
    {
        std::shared_ptr<armnn::IProfiler> profiler = m_Runtime->GetProfiler(m_NetworkId);

        std::shared_ptr<armnn::AsyncExecutionCallback> cb = callbackManager.GetNewCallback();
        inferenceOutputMap.insert({cb->GetInferenceId(), &m_OutputTensorsVec[i]});
        threadpool->Schedule(m_NetworkId,
                             m_InputTensorsVec[i],
                             m_OutputTensorsVec[i],
                             armnn::QosExecPriority::Medium,
                             cb);
    }

    // Check the results
    for (size_t iteration = 0; iteration < m_Params.m_Iterations; ++iteration)
    {
        auto cb = callbackManager.GetNotifiedCallback();

        // Get the results
        if (earliestStartTime > cb->GetStartTime())
        {
            earliestStartTime = cb->GetStartTime();
        }
        if (latestEndTime < cb->GetEndTime())
        {
            latestEndTime = cb->GetEndTime();
        }

        auto startTime = time_point_cast<std::chrono::milliseconds>(cb->GetStartTime());
        auto endTime = time_point_cast<std::chrono::milliseconds>(cb->GetEndTime());
        auto inferenceDuration = endTime - startTime;
        CheckInferenceTimeThreshold(inferenceDuration, m_Params.m_ThresholdTime);
        if(!m_Params.m_DontPrintOutputs)
        {
            const armnn::OutputTensors* out = inferenceOutputMap[cb->GetInferenceId()];
            PrintOutputTensors(out, iteration);
        }
    }

    // Print duration difference between overallStartTime and overallEndTime
    auto overallEndTime = time_point_cast<std::chrono::milliseconds>(latestEndTime);
    auto overallStartTime = time_point_cast<std::chrono::milliseconds>(earliestStartTime);
    auto totalInferenceDuration = overallEndTime - overallStartTime;
    ARMNN_LOG(info) << "Overall Inference time: " << std::setprecision(2)
                    << std::fixed << totalInferenceDuration.count() << " ms\n";

#endif
}

void ArmNNExecutor::ExecuteSync()
{
    for (size_t x = 0; x < m_Params.m_Iterations; x++)
    {
        std::shared_ptr<armnn::IProfiler> profiler = m_Runtime->GetProfiler(m_NetworkId);

        const auto start_time = armnn::GetTimeNow();
        armnn::Status ret;
        if (m_Params.m_ImportInputsIfAligned)
        {
             ret = m_Runtime->EnqueueWorkload(m_NetworkId,
                                              m_InputTensorsVec[x],
                                              m_OutputTensorsVec[x],
                                              m_ImportedInputIds[x],
                                              m_ImportedOutputIds[x]);
        }
        else
        {
            ret = m_Runtime->EnqueueWorkload(m_NetworkId,
                                             m_InputTensorsVec[x],
                                             m_OutputTensorsVec[x]);
        }

        const auto inferenceDuration = armnn::GetTimeDuration(start_time);

        // If profiling is enabled print out the results
        if(profiler && profiler->IsProfilingEnabled() && x == (m_Params.m_Iterations - 1))
        {
            profiler->Print(std::cout);
        }

        if(ret == armnn::Status::Failure)
        {
            throw armnn::Exception("IRuntime::EnqueueWorkload failed");
        }

        if(!m_Params.m_DontPrintOutputs)
        {
            PrintOutputTensors(&m_OutputTensorsVec[x],  x);
        }

        // If thresholdTime == 0.0 (default), then it hasn't been supplied at command line
        CheckInferenceTimeThreshold(inferenceDuration, m_Params.m_ThresholdTime);
    }
}

std::vector<const void*> ArmNNExecutor::Execute()
{
    if(m_Params.m_ThreadPoolSize == 0)
    {
        ExecuteSync();
    }
    else
    {
        ExecuteAsync();
    }
    std::vector<const void*> results;
    for (auto& output : m_OutputStorage)
    {
        results.push_back(output.m_Mem);
    }

    return results;
}

void ArmNNExecutor::PrintNetworkInfo()
{
    const std::vector<std::string>& inputNames = m_Params.m_InputNames.size() != 0 ?
                                                 m_Params.m_InputNames :
                                                 m_IOInfo.m_InputNames;
    std::stringstream ss;
    ss << "===== Network Info =====\n";
    ss << "Inputs in order:\n";
    for (const auto& inputName : inputNames)
    {
        const auto inputInfo = m_IOInfo.m_InputInfoMap[inputName].second;
        ss <<  inputName << ", " << inputInfo.GetShape() << ", " << GetDataTypeName(inputInfo.GetDataType());
        if (inputInfo.IsQuantized())
        {
            ss << " Quantization Offset: " << inputInfo.GetQuantizationOffset();
            if (inputInfo.HasMultipleQuantizationScales())
            {
                ss << " Quantization scales: ";
                for (const auto scale: inputInfo.GetQuantizationScales())
                {
                    ss << scale << ", ";
                }
            }
            else
            {
                ss << " Quantization scale: " << inputInfo.GetQuantizationScale();
            }
        }
        ss  << "\n";
    }

    ss << "Outputs in order:\n";
    for (const auto& outputName : m_IOInfo.m_OutputNames)
    {
        const auto outputInfo = m_IOInfo.m_OutputInfoMap[outputName].second;
        ss <<  outputName << ", " << outputInfo.GetShape() << ", " << GetDataTypeName(outputInfo.GetDataType());
        if (outputInfo.IsQuantized())
        {
            ss << " Quantization Offset: " << outputInfo.GetQuantizationOffset();
            if (outputInfo.HasMultipleQuantizationScales())
            {
                ss << " Quantization scales: ";
                for (const auto scale: outputInfo.GetQuantizationScales())
                {
                    ss << scale << ", ";
                }
            }
            else
            {
                ss << " Quantization scale: " << outputInfo.GetQuantizationScale();
            }
        }
        ss  << "\n";
    }

    std::cout << ss.str() << std::endl;
}

void ArmNNExecutor::SetupInputsAndOutputs()
{
    const unsigned int noOfInputs = m_IOInfo.m_InputNames.size();

    if (m_Params.m_InputNames.size() != 0 && m_Params.m_InputNames.size() != noOfInputs)
    {
        LogAndThrow("Number of input names does not match number of inputs");
    }

    const unsigned int inputFilePaths = m_Params.m_InputTensorDataFilePaths.size();
    const std::vector<std::string>& inputNames = m_Params.m_InputNames.size() != 0 ?
                                                 m_Params.m_InputNames :
                                                 m_IOInfo.m_InputNames;
    unsigned int noInputSets = 1;

    if (inputFilePaths != 0)
    {
        if (inputFilePaths % noOfInputs != 0)
        {
            LogAndThrow("Number of input files: " + std::to_string(inputFilePaths) +
                        " not compatible with number of inputs: " + std::to_string(noOfInputs));
        }
        noInputSets = inputFilePaths / noOfInputs;
        if (noInputSets != 1 && m_Params.m_ReuseBuffers)
        {
            LogAndThrow("Specifying multiple sets of inputs not compatible with ReuseBuffers");
        }
    }

    const unsigned int noOfOutputs = m_IOInfo.m_OutputNames.size();
    const unsigned int outputFilePaths = m_Params.m_OutputTensorFiles.size();
    unsigned int noOutputSets = 1;

    if (outputFilePaths != 0)
    {
        if (outputFilePaths % noOfOutputs != 0)
        {
            LogAndThrow("Number of output files: " + std::to_string(outputFilePaths) +
                        ", not compatible with number of outputs: " + std::to_string(noOfOutputs));
        }
        noOutputSets = outputFilePaths / noOfOutputs;

        if (noOutputSets != 1 && m_Params.m_ReuseBuffers)
        {
            LogAndThrow("Specifying multiple sets of outputs not compatible with ReuseBuffers");
        }
    }

    if (m_Params.m_ThreadPoolSize != 0)
    {
        // The current implementation of the Threadpool does not allow binding of outputs to a thread
        // So to ensure no two threads write to the same output at the same time, no output can be reused
        noOutputSets = m_Params.m_Iterations;
    }

    if (m_Params.m_InputTensorDataFilePaths.size() > noOfInputs)
    {
        ARMNN_LOG(info) << "Given network has " << noOfInputs << " input/s. One input-tensor-data file is required "
                        << "for each input. The user provided "
                        << m_Params.m_InputTensorDataFilePaths.size()
                        << " input-tensor-data file/s which will be used to fill the input/s.\n";
    }

    unsigned int inputCount = 0;
    for(unsigned int inputSet = 0; inputSet < noInputSets; ++inputSet)
    {
        armnn::InputTensors inputTensors;
        for (const auto& inputName: inputNames)
        {
            armnn::BindingPointInfo bindingPointInfo;
            try
            {
                bindingPointInfo = m_IOInfo.m_InputInfoMap.at(inputName);
            }
            catch (const std::out_of_range& e)
            {
                LogAndThrow("Input with inputName: " + inputName + " not found.");
            }

            const armnn::TensorInfo& tensorInfo = bindingPointInfo.second;
            auto newInfo = armnn::TensorInfo{tensorInfo.GetShape(), tensorInfo.GetDataType(),
                                             tensorInfo.GetQuantizationScale(),
                                             tensorInfo.GetQuantizationOffset(),
                                             true};

            m_InputStorage.emplace_back(IOStorage{tensorInfo.GetNumBytes()});

            const int bindingId = bindingPointInfo.first;
            inputTensors.emplace_back(bindingId, armnn::ConstTensor{newInfo, m_InputStorage.back().m_Mem});

            const armnn::Optional<std::string> dataFile = m_Params.m_GenerateTensorData ?
                                                          armnn::EmptyOptional() :
                                                          armnn::MakeOptional<std::string>(
                                                                  m_Params.m_InputTensorDataFilePaths.at(inputCount++));

            switch (tensorInfo.GetDataType())
            {
                case armnn::DataType::Float32:
                {
                    auto typedTensor = reinterpret_cast<float*>(m_InputStorage.back().m_Mem);
                    PopulateTensorWithData<float>(typedTensor, tensorInfo.GetNumElements(), dataFile, inputName);
                    break;
                }
                case armnn::DataType::QSymmS16:
                {
                    auto typedTensor = reinterpret_cast<int16_t*>(m_InputStorage.back().m_Mem);
                    PopulateTensorWithData<int16_t>(typedTensor, tensorInfo.GetNumElements(), dataFile, inputName);
                    break;
                }
                case armnn::DataType::QSymmS8:
                case armnn::DataType::QAsymmS8:
                {
                    auto typedTensor = reinterpret_cast<int8_t*>(m_InputStorage.back().m_Mem);
                    PopulateTensorWithData<int8_t>(typedTensor, tensorInfo.GetNumElements(), dataFile, inputName);
                    break;
                }
                case armnn::DataType::QAsymmU8:
                {
                    auto typedTensor = reinterpret_cast<uint8_t*>(m_InputStorage.back().m_Mem);
                    PopulateTensorWithData<uint8_t>(typedTensor, tensorInfo.GetNumElements(), dataFile, inputName);
                    break;
                }
                case armnn::DataType::Signed32:
                {
                    auto typedTensor = reinterpret_cast<int32_t*>(m_InputStorage.back().m_Mem);
                    PopulateTensorWithData<int32_t>(typedTensor, tensorInfo.GetNumElements(), dataFile, inputName);
                    break;
                }
                default:
                {
                    LogAndThrow("Unexpected DataType");
                }
            }

        }

        if (m_Params.m_ImportInputsIfAligned)
        {
            m_ImportedInputIds.push_back(
                m_Runtime->ImportInputs(m_NetworkId, inputTensors, armnn::MemorySource::Malloc));
        }
        m_InputTensorsVec.emplace_back(inputTensors);
    }

    for(unsigned int outputSet = 0; outputSet < noOutputSets; ++outputSet)
    {
        armnn::OutputTensors outputTensors;
        for (const auto& output: m_IOInfo.m_OutputInfoMap)
        {
            const armnn::BindingPointInfo& bindingPointInfo = output.second;
            const armnn::TensorInfo& tensorInfo = bindingPointInfo.second;

            m_OutputStorage.emplace_back(tensorInfo.GetNumBytes());
            outputTensors.emplace_back(bindingPointInfo.first, armnn::Tensor{tensorInfo, m_OutputStorage.back().m_Mem});
        }
        m_OutputTensorsVec.emplace_back(outputTensors);
        if (m_Params.m_ImportInputsIfAligned)
        {
            m_ImportedOutputIds.push_back(
                    m_Runtime->ImportOutputs(m_NetworkId, m_OutputTensorsVec.back(), armnn::MemorySource::Malloc));
        }
    }

    // If iterations > noSets fill the remaining iterations repeating the given files
    // If iterations < noSets just ignore the extra files
    const unsigned int remainingInputSets = (m_Params.m_Iterations > noInputSets)
                                          ? m_Params.m_Iterations - noInputSets
                                          : 0;
    for (unsigned int i = 0; i < remainingInputSets; ++i)
    {
        m_InputTensorsVec.push_back(m_InputTensorsVec[i % noInputSets]);
        if (m_Params.m_ImportInputsIfAligned)
        {
            m_ImportedInputIds.push_back(m_ImportedInputIds[i % noInputSets]);
        }
    }

    const unsigned int remainingOutputSets = (m_Params.m_Iterations > noOutputSets)
                                           ? m_Params.m_Iterations - noOutputSets
                                           : 0;
    for (unsigned int i = 0; i < remainingOutputSets; ++i)
    {
        m_OutputTensorsVec.push_back(m_OutputTensorsVec[i % noOutputSets]);
        if (m_Params.m_ImportInputsIfAligned)
        {
            m_ImportedOutputIds.push_back(m_ImportedOutputIds[i % noOutputSets]);
        }
    }
}

ArmNNExecutor::IOInfo ArmNNExecutor::GetIOInfo(armnn::IOptimizedNetwork* optNet)
{
    struct IOStrategy : armnn::IStrategy
    {
        void ExecuteStrategy(const armnn::IConnectableLayer* layer,
                             const armnn::BaseDescriptor& descriptor,
                             const std::vector<armnn::ConstTensor>& constants,
                             const char* name,
                             const armnn::LayerBindingId id = 0) override
        {
            armnn::IgnoreUnused(descriptor, constants, id);
            switch (layer->GetType())
            {
                case armnn::LayerType::Input:
                {
                    m_IOInfo.m_InputNames.emplace_back(name);
                    m_IOInfo.m_InputInfoMap[name] = {id, layer->GetOutputSlot(0).GetTensorInfo()};
                    break;
                }
                case armnn::LayerType::Output:
                {
                    m_IOInfo.m_OutputNames.emplace_back(name);
                    m_IOInfo.m_OutputInfoMap[name] = {id, layer->GetInputSlot(0).GetConnection()->GetTensorInfo()};
                    break;
                }
                default: {}
            }
        }
        IOInfo m_IOInfo;
    };

    IOStrategy ioStrategy;
    optNet->ExecuteStrategy(ioStrategy);

    return ioStrategy.m_IOInfo;
}

armnn::IOptimizedNetworkPtr ArmNNExecutor::OptimizeNetwork(armnn::INetwork* network)
{
    armnn::IOptimizedNetworkPtr optNet{nullptr, [](armnn::IOptimizedNetwork*){}};

    armnn::OptimizerOptionsOpaque options;
    options.SetReduceFp32ToFp16(m_Params.m_EnableFp16TurboMode);
    options.SetDebugEnabled(m_Params.m_PrintIntermediate);
    options.SetDebugToFileEnabled(m_Params.m_PrintIntermediateOutputsToFile);
    options.SetShapeInferenceMethod(m_Params.m_InferOutputShape ?
                                    armnn::ShapeInferenceMethod::InferAndValidate :
                                    armnn::ShapeInferenceMethod::ValidateOnly);
    options.SetProfilingEnabled(m_Params.m_EnableProfiling);
    options.SetAllowExpandedDims(m_Params.m_AllowExpandedDims);

    armnn::BackendOptions gpuAcc("GpuAcc",
                                 {
                                         { "FastMathEnabled", m_Params.m_EnableFastMath },
                                         { "SaveCachedNetwork", m_Params.m_SaveCachedNetwork },
                                         { "CachedNetworkFilePath", m_Params.m_CachedNetworkFilePath },
                                         { "MLGOTuningFilePath", m_Params.m_MLGOTuningFilePath }
                                 });

    armnn::BackendOptions cpuAcc("CpuAcc",
                                 {
                                         { "FastMathEnabled", m_Params.m_EnableFastMath },
                                         { "NumberOfThreads", m_Params.m_NumberOfThreads }
                                 });
    options.AddModelOption(gpuAcc);
    options.AddModelOption(cpuAcc);
    // The shapeInferenceMethod and allowExpandedDims values have to be added to the model options
    // because these are what are passed to the OptimizeSubgraphViews method and are used to create
    // the new optimized INetwork that method uses
    armnn::BackendOptions allowExDimOpt("AllowExpandedDims",
                                        {
                                                { "AllowExpandedDims", m_Params.m_AllowExpandedDims }
                                        });
    options.AddModelOption(allowExDimOpt);
    armnn::BackendOptions shapeInferOpt("ShapeInferenceMethod",
                                        {
                                                { "InferAndValidate", m_Params.m_InferOutputShape }
                                        });
    options.AddModelOption(shapeInferOpt);

    const auto optimization_start_time = armnn::GetTimeNow();
    optNet = armnn::Optimize(*network, m_Params.m_ComputeDevices, m_Runtime->GetDeviceSpec(), options);

    ARMNN_LOG(info) << "Optimization time: " << std::setprecision(2)
                    << std::fixed << armnn::GetTimeDuration(optimization_start_time).count() << " ms\n";

    if (!optNet)
    {
        LogAndThrow("Optimize returned nullptr");
    }

    // If v,visualize-optimized-model is enabled then construct a file name for the dot file.
    if (m_Params.m_EnableLayerDetails)
    {
        fs::path filename = m_Params.m_ModelPath;
        filename.replace_extension("dot");
        std::fstream file(filename.c_str(), std::ios_base::out);
        optNet->SerializeToDot(file);
    }

    return optNet;
}

std::unique_ptr<ArmNNExecutor::IParser> ArmNNExecutor::CreateParser()
{
    const fs::path modelFilename = m_Params.m_ModelPath;
    const std::string modelExtension = modelFilename.extension();

    m_Params.m_IsModelBinary = modelExtension != ".json";
    std::unique_ptr<IParser> parser = nullptr;
    // Forward to implementation based on the parser type
    if (modelExtension == ".armnn")
    {
#if defined(ARMNN_SERIALIZER)
        parser = std::make_unique<ArmNNDeserializer>();
#else
        LogAndThrow("Not built with serialization support.");
#endif
    }
    else if (modelExtension == ".tflite")
    {
#if defined(ARMNN_TF_LITE_PARSER)
        parser = std::make_unique<TfliteParser>(m_Params);
#else
        LogAndThrow("Not built with Tensorflow-Lite parser support.");
#endif
    }
    else if (modelExtension == ".onnx")
    {
#if defined(ARMNN_ONNX_PARSER)
        parser = std::make_unique<OnnxParser>();
#else
        LogAndThrow("Not built with Onnx parser support.");
#endif
    }

    return parser;
}

void ArmNNExecutor::PrintOutputTensors(const armnn::OutputTensors* outputTensors,
                                       unsigned int iteration)
{
    auto findOutputName = [&](const armnn::LayerBindingId id)
    {
        for (auto it = m_IOInfo.m_OutputInfoMap.begin(); it != m_IOInfo.m_OutputInfoMap.end(); ++it)
        {
            if (id == it->second.first)
            {
                return it->first;
            }
        }
        return std::string{};
    };

    unsigned int outputIndex = 0;
    unsigned int numOutputs = outputTensors->size();
    for (const auto& output: *outputTensors)
    {
        const auto bindingName = findOutputName(output.first);
        // We've made sure before that the number of output files either equals numOutputs, in which
        // case we override those files when processing the results of each iteration (only the result
        // of the last iteration will be stored), or there are enough
        // output files for each output of each iteration.
        size_t outputFileIndex = iteration * numOutputs + outputIndex;
        if (!m_Params.m_OutputTensorFiles.empty())
        {
            outputFileIndex = outputFileIndex % m_Params.m_OutputTensorFiles.size();
            ARMNN_LOG(info) << "Writing output: " << bindingName << " bindingId: '"
                            << output.first
                            << "' of iteration: " << iteration + 1 << " to file: '"
                            << m_Params.m_OutputTensorFiles[outputFileIndex] << "'";
        }

        const armnn::Optional<std::string> outputTensorFile = m_Params.m_OutputTensorFiles.empty() ?
                                                              armnn::EmptyOptional() :
                                                              armnn::MakeOptional<std::string>(
                                                                      m_Params.m_OutputTensorFiles[outputFileIndex]);

        OutputWriteInfo outputWriteInfo
        {
            outputTensorFile,
            bindingName,
            output.second,
            !m_Params.m_DontPrintOutputs
        };

        std::cout << bindingName << ": ";
        std::vector<float> values;
        switch (output.second.GetDataType())
        {
            case armnn::DataType::Float32:
            {
                PrintTensor<float>(outputWriteInfo, "%f ");
                break;
            }

            case armnn::DataType::Signed32:
            {
                PrintTensor<int>(outputWriteInfo, "%d ");
                break;
            }
            case armnn::DataType::QSymmS8:
            case armnn::DataType::QAsymmS8:
            {
                PrintTensor<int8_t>(outputWriteInfo, "%d ");
                break;
            }
            case armnn::DataType::QAsymmU8:
            {
                PrintTensor<uint8_t>(outputWriteInfo, "%d ");
                break;
            }
            case armnn::DataType::Float16:
            case armnn::DataType::QSymmS16:
            case armnn::DataType::BFloat16:
            case armnn::DataType::Boolean:
            case armnn::DataType::Signed64:
            default:
            {
                LogAndThrow("Unexpected DataType");
            }
        }
        std::cout << "\n";
        ++outputIndex;
    }
}

void ArmNNExecutor::CompareAndPrintResult(std::vector<const void*> otherOutput)
{
    unsigned int index = 0;
    std::string typeString;
    for (const auto& outputTensors: m_OutputTensorsVec)
    {
        for (const auto& outputTensor: outputTensors)
        {
            size_t size = outputTensor.second.GetNumBytes();
            double result = ComputeByteLevelRMSE(outputTensor.second.GetMemoryArea(), otherOutput[index++], size);
            std::cout << "Byte level root mean square error: " << result << "\n";
        }
    }
}
#if defined(ARMNN_SERIALIZER)
ArmNNExecutor::ArmNNDeserializer::ArmNNDeserializer() : m_Parser(armnnDeserializer::IDeserializer::Create()){}

armnn::INetworkPtr ArmNNExecutor::ArmNNDeserializer::CreateNetwork(const ExecuteNetworkParams& params)
{
    const std::string& modelPath = params.m_ModelPath;

    std::ifstream file(modelPath, std::ios::binary);
    return m_Parser->CreateNetworkFromBinary(file);
}

armnn::BindingPointInfo
ArmNNExecutor::ArmNNDeserializer::GetInputBindingPointInfo(size_t, const std::string& inputName)
{
    armnnDeserializer::BindingPointInfo DeserializerBPI = m_Parser->GetNetworkInputBindingInfo(0, inputName);
    return {DeserializerBPI.m_BindingId, DeserializerBPI.m_TensorInfo};
}

armnn::BindingPointInfo
ArmNNExecutor::ArmNNDeserializer::GetOutputBindingPointInfo(size_t, const std::string& outputName)
{
    armnnDeserializer::BindingPointInfo DeserializerBPI = m_Parser->GetNetworkOutputBindingInfo(0, outputName);
    return {DeserializerBPI.m_BindingId, DeserializerBPI.m_TensorInfo};
}
#endif

#if defined(ARMNN_TF_LITE_PARSER)
ArmNNExecutor::TfliteParser::TfliteParser(const ExecuteNetworkParams& params)
{
    armnnTfLiteParser::ITfLiteParser::TfLiteParserOptions options;
    options.m_StandInLayerForUnsupported = params.m_ParseUnsupported;
    options.m_InferAndValidate = params.m_InferOutputShape;
    options.m_AllowExpandedDims = params.m_AllowExpandedDims;

    m_Parser = armnnTfLiteParser::ITfLiteParser::Create(options);
}

armnn::INetworkPtr ArmNNExecutor::TfliteParser::CreateNetwork(const ExecuteNetworkParams& params)
{
    const std::string& modelPath = params.m_ModelPath;
    return m_Parser->CreateNetworkFromBinaryFile(modelPath.c_str());
}

armnn::BindingPointInfo ArmNNExecutor::TfliteParser::GetInputBindingPointInfo(size_t subgraphId,
                                                                              const std::string& inputName)
{
    return m_Parser->GetNetworkInputBindingInfo(subgraphId, inputName);
}

armnn::BindingPointInfo ArmNNExecutor::TfliteParser::GetOutputBindingPointInfo(size_t subgraphId,
                                                                               const std::string& outputName)
{
    return m_Parser->GetNetworkOutputBindingInfo(subgraphId, outputName);
}
#endif


#if defined(ARMNN_ONNX_PARSER)
ArmNNExecutor::OnnxParser::OnnxParser() : m_Parser(armnnOnnxParser::IOnnxParser::Create()){}

armnn::INetworkPtr ArmNNExecutor::OnnxParser::CreateNetwork(const ExecuteNetworkParams& params)
{
    const std::string& modelPath = params.m_ModelPath;
    m_Parser = armnnOnnxParser::IOnnxParser::Create();
    std::map<std::string, armnn::TensorShape> inputShapes;
    if(!params.m_InputTensorShapes.empty())
    {
        const size_t numInputShapes = params.m_InputTensorShapes.size();
        const size_t numInputBindings = params.m_InputNames.size();
        if(numInputShapes < numInputBindings)
        {
            throw armnn::Exception(
                    fmt::format("Not every input has its tensor shape specified: expected={0}, got={1}",
                                numInputBindings, numInputShapes));
        }

        for (size_t i = 0; i < numInputShapes; i++)
        {
            inputShapes[params.m_InputNames[i]] = params.m_InputTensorShapes[i];
        }

        return params.m_IsModelBinary ?
               m_Parser->CreateNetworkFromBinaryFile(modelPath.c_str(), inputShapes) :
               m_Parser->CreateNetworkFromTextFile(modelPath.c_str(), inputShapes);
    }

    // Handle text and binary input differently by calling the corresponding parser function
    return params.m_IsModelBinary ?
           m_Parser->CreateNetworkFromBinaryFile(params.m_ModelPath.c_str()) :
           m_Parser->CreateNetworkFromTextFile(params.m_ModelPath.c_str());
}

armnn::BindingPointInfo ArmNNExecutor::OnnxParser::GetInputBindingPointInfo(size_t, const std::string& inputName)
{
    return m_Parser->GetNetworkInputBindingInfo(inputName);
}

armnn::BindingPointInfo ArmNNExecutor::OnnxParser::GetOutputBindingPointInfo(size_t, const std::string& outputName)
{
    return m_Parser->GetNetworkOutputBindingInfo(outputName);
}
#endif
