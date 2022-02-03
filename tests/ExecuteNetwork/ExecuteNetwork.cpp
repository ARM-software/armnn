//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NetworkExecutionUtils/NetworkExecutionUtils.hpp"
#include "ExecuteNetworkProgramOptions.hpp"
#include <armnn/IAsyncExecutionCallback.hpp>
#include <AsyncExecutionCallback.hpp>

#include <armnn/Logging.hpp>
#include <armnnUtils/Filesystem.hpp>
#include <armnnUtils/TContainer.hpp>
#include <InferenceTest.hpp>

#if defined(ARMNN_SERIALIZER)
#include "armnnDeserializer/IDeserializer.hpp"
#endif
#if defined(ARMNN_TF_LITE_PARSER)
#include "armnnTfLiteParser/ITfLiteParser.hpp"
#endif
#if defined(ARMNN_ONNX_PARSER)
#include "armnnOnnxParser/IOnnxParser.hpp"
#endif
#if defined(ARMNN_TFLITE_DELEGATE)
#include <armnn_delegate.hpp>
#include <DelegateOptions.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/kernels/builtin_op_kernels.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#endif

#include <future>

/**
 * Given a measured duration and a threshold time tell the user whether we succeeded or not.
 *
 * @param duration the measured inference duration.
 * @param thresholdTime the threshold time in milliseconds.
 * @return false if the measured time exceeded the threshold.
 */
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

#if defined(ARMNN_TFLITE_DELEGATE)
int TfLiteDelegateMainImpl(const ExecuteNetworkParams& params, const armnn::IRuntime::CreationOptions runtimeOptions)
{
    // Build model and corresponding interpreter
    using namespace tflite;

    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(params.m_ModelPath.c_str());

    auto tfLiteInterpreter =  std::make_unique<Interpreter>();
    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&tfLiteInterpreter);
    tfLiteInterpreter->AllocateTensors();

    int status = 0;

    // Create & populate Armnn Delegate, then register it to TfLiteInterpreter
    if (params.m_TfLiteExecutor == ExecuteNetworkParams::TfLiteExecutor::ArmNNTfLiteDelegate)
    {
        // Create the Armnn Delegate
        // Populate a DelegateOptions from the ExecuteNetworkParams.
        armnnDelegate::DelegateOptions delegateOptions = params.ToDelegateOptions();
        delegateOptions.SetExternalProfilingParams(runtimeOptions.m_ProfilingOptions);

        std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
                theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                                 armnnDelegate::TfLiteArmnnDelegateDelete);
        // Register armnn_delegate to TfLiteInterpreter
        status = tfLiteInterpreter->ModifyGraphWithDelegate(std::move(theArmnnDelegate));
        if (status != kTfLiteOk)
        {
            ARMNN_LOG(fatal) << "Could not register ArmNN TfLite Delegate to TfLiteInterpreter!";
            return EXIT_FAILURE;
        }
    }
    else
    {
        std::cout << "Running on TfLite without ArmNN delegate\n";
    }

    // Load (or generate) input data for inference
    armnn::Optional<std::string> dataFile = params.m_GenerateTensorData
                                            ? armnn::EmptyOptional()
                                            : armnn::MakeOptional<std::string>(params.m_InputTensorDataFilePaths[0]);

    const size_t numInputs = params.m_InputNames.size();

    // Populate input tensor of interpreter
    for(unsigned int inputIndex = 0; inputIndex < numInputs; ++inputIndex)
    {
        int input = tfLiteInterpreter->inputs()[inputIndex];
        TfLiteIntArray* inputDims = tfLiteInterpreter->tensor(input)->dims;

        unsigned int inputSize = 1;
        if (params.m_InputTensorShapes.size() > 0)
        {
            inputSize = params.m_InputTensorShapes[inputIndex]->GetNumElements();
        }
        else
        {
            for (unsigned int dim = 0; dim < static_cast<unsigned int>(inputDims->size); ++dim)
            {
                inputSize *= inputDims->data[dim];
            }
        }

        if (params.m_InputTypes[inputIndex].compare("float") == 0)
        {
            auto inputData = tfLiteInterpreter->typed_tensor<float>(input);

            if(inputData == NULL)
            {
                ARMNN_LOG(fatal) << "Input tensor is null, input type: "
                                    "\"" << params.m_InputTypes[inputIndex] << "\" may be incorrect.";
                return EXIT_FAILURE;
            }

            std::vector<float> tensorData;
            PopulateTensorWithDataGeneric<float>(tensorData,
                                                 inputSize,
                                                 dataFile,
                                                 [](const std::string& s)
                                                 { return std::stof(s); });

            std::copy(tensorData.begin(), tensorData.end(), inputData);
        }
        else if (params.m_InputTypes[inputIndex].compare("qsymms8") == 0 ||
                 params.m_InputTypes[inputIndex].compare("qasymms8") == 0)
        {
            auto inputData = tfLiteInterpreter->typed_tensor<int8_t>(input);

            if(inputData == NULL)
            {
                ARMNN_LOG(fatal) << "Input tensor is null, input type: "
                                    "\"" << params.m_InputTypes[inputIndex] << "\" may be incorrect.";
                return EXIT_FAILURE;
            }

            std::vector<int8_t> tensorData;
            PopulateTensorWithDataGeneric<int8_t>(tensorData,
                                                  inputSize,
                                                  dataFile,
                                                  [](const std::string& s)
                                                  { return armnn::numeric_cast<int8_t>(std::stoi(s)); });

            std::copy(tensorData.begin(), tensorData.end(), inputData);
        }
        else if (params.m_InputTypes[inputIndex].compare("int") == 0)
        {
            auto inputData = tfLiteInterpreter->typed_tensor<int32_t>(input);

            if(inputData == NULL)
            {
                ARMNN_LOG(fatal) << "Input tensor is null, input type: "
                                    "\"" << params.m_InputTypes[inputIndex] << "\" may be incorrect.";
                return EXIT_FAILURE;
            }

            std::vector<int32_t> tensorData;
            PopulateTensorWithDataGeneric<int32_t>(tensorData,
                                                   inputSize,
                                                   dataFile,
                                                   [](const std::string& s)
                                                   { return std::stoi(s); });

            std::copy(tensorData.begin(), tensorData.end(), inputData);
        }
        else if (params.m_InputTypes[inputIndex].compare("qasymm8") == 0 ||
                 params.m_InputTypes[inputIndex].compare("qasymmu8") == 0)
        {
            auto inputData = tfLiteInterpreter->typed_tensor<uint8_t>(input);

            if(inputData == NULL)
            {
                ARMNN_LOG(fatal) << "Input tensor is null, input type: "
                                    "\"" << params.m_InputTypes[inputIndex] << "\" may be incorrect.";
                return EXIT_FAILURE;
            }

            std::vector<uint8_t> tensorData;
            PopulateTensorWithDataGeneric<uint8_t>(tensorData,
                                                   inputSize,
                                                   dataFile,
                                                   [](const std::string& s)
                                                   { return armnn::numeric_cast<uint8_t>(std::stoi(s)); });

            std::copy(tensorData.begin(), tensorData.end(), inputData);
        }
        else
        {
            ARMNN_LOG(fatal) << "Unsupported input tensor data type \"" << params.m_InputTypes[inputIndex] << "\". ";
            return EXIT_FAILURE;
        }
    }

    // Run inference, print the output of the inference
    for (size_t x = 0; x < params.m_Iterations; x++)
    {
        // Start timer to record inference time in milliseconds.
        const auto start_time = armnn::GetTimeNow();
        // Run the inference
        status = tfLiteInterpreter->Invoke();
        const auto duration = armnn::GetTimeDuration(start_time);

        // The TFLite interpreter's outputs might be in a different order than the user inputted output names.
        std::map<unsigned int, int> paramToTfliteOutputIndex;
        for (unsigned int paramIndex = 0; paramIndex < params.m_OutputNames.size(); ++paramIndex)
        {
            paramToTfliteOutputIndex[paramIndex] = -1;
            for (unsigned int tfLiteIndex = 0; tfLiteIndex < tfLiteInterpreter->outputs().size(); ++tfLiteIndex)
            {
                if (params.m_OutputNames[paramIndex] == tfLiteInterpreter->GetOutputName(tfLiteIndex))
                {
                    paramToTfliteOutputIndex[paramIndex] = tfLiteIndex;
                }
            }
        }

        // Print out the output
        for (unsigned int paramOutputIndex = 0; paramOutputIndex < params.m_OutputNames.size(); ++paramOutputIndex)
        {
            int outputIndex = paramToTfliteOutputIndex[paramOutputIndex];
            if (outputIndex == -1)
            {
                std::cout << fmt::format("Output name: {} doesn't exist.", params.m_OutputNames[paramOutputIndex]) <<
                std::endl;
                continue;
            }
            auto tfLiteDelegateOutputId = tfLiteInterpreter->outputs()[outputIndex];
            TfLiteIntArray* outputDims = tfLiteInterpreter->tensor(tfLiteDelegateOutputId)->dims;
            // If we've been asked to write to a file then set a file output stream. Otherwise use stdout.
            FILE* outputTensorFile = stdout;
            if (!params.m_OutputTensorFiles.empty())
            {
                outputTensorFile = fopen(params.m_OutputTensorFiles[outputIndex].c_str(), "w");
                if (outputTensorFile == NULL)
                {
                    ARMNN_LOG(fatal) << "Specified output tensor file, \"" <<
                                     params.m_OutputTensorFiles[outputIndex] <<
                                     "\", cannot be created. Defaulting to stdout. " <<
                                     "Error was: " << std::strerror(errno);
                    outputTensorFile = stdout;
                }
                else
                {
                    ARMNN_LOG(info) << "Writing output " << outputIndex << "' of iteration: " << x+1 << " to file: '"
                                    << params.m_OutputTensorFiles[outputIndex] << "'";
                }
            }
            long outputSize = 1;
            for (unsigned int dim = 0; dim < static_cast<unsigned int>(outputDims->size); ++dim)
            {
                outputSize *=  outputDims->data[dim];
            }

            std::cout << tfLiteInterpreter->GetOutputName(outputIndex) << ": ";
            if (params.m_OutputTypes[paramOutputIndex].compare("float") == 0)
            {
                auto tfLiteDelageOutputData = tfLiteInterpreter->typed_tensor<float>(tfLiteDelegateOutputId);
                if(tfLiteDelageOutputData == NULL)
                {
                    ARMNN_LOG(fatal) << "Output tensor is null, output type: "
                                        "\"" << params.m_OutputTypes[paramOutputIndex] << "\" may be incorrect.";
                    return EXIT_FAILURE;
                }

                if (!params.m_DontPrintOutputs)
                {
                    for (int i = 0; i < outputSize; ++i)
                    {
                        fprintf(outputTensorFile, "%f ", tfLiteDelageOutputData[i]);
                    }
                }
            }
            else if (params.m_OutputTypes[paramOutputIndex].compare("int") == 0)
            {
                auto tfLiteDelageOutputData = tfLiteInterpreter->typed_tensor<int32_t>(tfLiteDelegateOutputId);
                if(tfLiteDelageOutputData == NULL)
                {
                    ARMNN_LOG(fatal) << "Output tensor is null, output type: "
                                        "\"" << params.m_OutputTypes[paramOutputIndex] << "\" may be incorrect.";
                    return EXIT_FAILURE;
                }

                if (!params.m_DontPrintOutputs)
                {
                    for (int i = 0; i < outputSize; ++i)
                    {
                        fprintf(outputTensorFile, "%d ", tfLiteDelageOutputData[i]);
                    }
                }
            }
            else if (params.m_OutputTypes[paramOutputIndex].compare("qsymms8") == 0 ||
                     params.m_OutputTypes[paramOutputIndex].compare("qasymms8") == 0)
            {
                auto tfLiteDelageOutputData = tfLiteInterpreter->typed_tensor<int8_t>(tfLiteDelegateOutputId);
                if(tfLiteDelageOutputData == NULL)
                {
                    ARMNN_LOG(fatal) << "Output tensor is null, output type: "
                                        "\"" << params.m_OutputTypes[paramOutputIndex] << "\" may be incorrect.";
                    return EXIT_FAILURE;
                }

                if (!params.m_DontPrintOutputs)
                {
                    for (int i = 0; i < outputSize; ++i)
                    {
                        fprintf(outputTensorFile, "%d ", tfLiteDelageOutputData[i]);
                    }
                }
            }
            else if (params.m_OutputTypes[paramOutputIndex].compare("qasymm8") == 0 ||
                     params.m_OutputTypes[paramOutputIndex].compare("qasymmu8") == 0)
            {
                auto tfLiteDelageOutputData = tfLiteInterpreter->typed_tensor<uint8_t>(tfLiteDelegateOutputId);
                if(tfLiteDelageOutputData == NULL)
                {
                    ARMNN_LOG(fatal) << "Output tensor is null, output type: "
                                        "\"" << params.m_OutputTypes[paramOutputIndex] << "\" may be incorrect.";
                    return EXIT_FAILURE;
                }

                if (!params.m_DontPrintOutputs)
                {
                    for (int i = 0; i < outputSize; ++i)
                    {
                        fprintf(outputTensorFile, "%u ", tfLiteDelageOutputData[i]);
                    }
                }
            }
            else
            {
                ARMNN_LOG(fatal) << "Output tensor is null, output type: "
                                    "\"" << params.m_OutputTypes[paramOutputIndex] <<
                                 "\" may be incorrect. Output type can be specified with -z argument";
                return EXIT_FAILURE;
            }
            std::cout << std::endl;
        }
        CheckInferenceTimeThreshold(duration, params.m_ThresholdTime);
    }

    return status;
}
#endif
template<typename TParser, typename TDataType>
int MainImpl(const ExecuteNetworkParams& params,
             const std::shared_ptr<armnn::IRuntime>& runtime = nullptr)
{
    using namespace std::chrono;

    std::vector<std::vector<armnnUtils::TContainer>> inputs;
    std::vector<std::vector<armnnUtils::TContainer>> outputs;

    try
    {
        // Creates an InferenceModel, which will parse the model and load it into an IRuntime.
        typename InferenceModel<TParser, TDataType>::Params inferenceModelParams;
        inferenceModelParams.m_ModelPath                      = params.m_ModelPath;
        inferenceModelParams.m_IsModelBinary                  = params.m_IsModelBinary;
        inferenceModelParams.m_ComputeDevices                 = params.m_ComputeDevices;
        inferenceModelParams.m_DynamicBackendsPath            = params.m_DynamicBackendsPath;
        inferenceModelParams.m_PrintIntermediateLayers        = params.m_PrintIntermediate;
        inferenceModelParams.m_VisualizePostOptimizationModel = params.m_EnableLayerDetails;
        inferenceModelParams.m_ParseUnsupported               = params.m_ParseUnsupported;
        inferenceModelParams.m_InferOutputShape               = params.m_InferOutputShape;
        inferenceModelParams.m_EnableFastMath                 = params.m_EnableFastMath;
        inferenceModelParams.m_SaveCachedNetwork              = params.m_SaveCachedNetwork;
        inferenceModelParams.m_CachedNetworkFilePath          = params.m_CachedNetworkFilePath;
        inferenceModelParams.m_NumberOfThreads                = params.m_NumberOfThreads;
        inferenceModelParams.m_MLGOTuningFilePath             = params.m_MLGOTuningFilePath;
        inferenceModelParams.m_AsyncEnabled                   = params.m_Concurrent;
        inferenceModelParams.m_ThreadPoolSize                 = params.m_ThreadPoolSize;
        inferenceModelParams.m_OutputDetailsToStdOut          = params.m_OutputDetailsToStdOut;
        inferenceModelParams.m_OutputDetailsOnlyToStdOut      = params.m_OutputDetailsOnlyToStdOut;

        for(const std::string& inputName: params.m_InputNames)
        {
            inferenceModelParams.m_InputBindings.push_back(inputName);
        }

        for(unsigned int i = 0; i < params.m_InputTensorShapes.size(); ++i)
        {
            inferenceModelParams.m_InputShapes.push_back(*params.m_InputTensorShapes[i]);
        }

        for(const std::string& outputName: params.m_OutputNames)
        {
            inferenceModelParams.m_OutputBindings.push_back(outputName);
        }

        inferenceModelParams.m_SubgraphId          = params.m_SubgraphId;
        inferenceModelParams.m_EnableFp16TurboMode = params.m_EnableFp16TurboMode;
        inferenceModelParams.m_EnableBf16TurboMode = params.m_EnableBf16TurboMode;

        InferenceModel<TParser, TDataType> model(inferenceModelParams,
                                                 params.m_EnableProfiling,
                                                 params.m_DynamicBackendsPath,
                                                 runtime);

        const size_t numInputs = inferenceModelParams.m_InputBindings.size();

        armnn::Optional<QuantizationParams> qParams = params.m_QuantizeInput ?
                                                      armnn::MakeOptional<QuantizationParams>(
                                                          model.GetInputQuantizationParams()) :
                                                      armnn::EmptyOptional();

        if (params.m_InputTensorDataFilePaths.size() > numInputs)
        {
            ARMNN_LOG(info) << "Given network has " << numInputs << " input/s. One input-tensor-data file is required "
                            << "for each input. The user provided "
                            << params.m_InputTensorDataFilePaths.size()
                            << " input-tensor-data file/s which will be used to fill the input/s.\n";
        }

        for(unsigned int j = 0; j < params.m_Iterations ; ++j)
        {
            std::vector<armnnUtils::TContainer> inputDataContainers;
            for(unsigned int i = 0; i < numInputs; ++i)
            {
                // If there are fewer input files given than required for the execution of
                // params.m_Iterations we simply start with the first input file again
                size_t inputFileIndex = j * numInputs + i;
                if (!params.m_InputTensorDataFilePaths.empty())
                {
                    inputFileIndex = inputFileIndex % params.m_InputTensorDataFilePaths.size();
                }

                armnn::Optional<std::string> dataFile = params.m_GenerateTensorData ?
                                                        armnn::EmptyOptional() :
                                                        armnn::MakeOptional<std::string>(
                                                            params.m_InputTensorDataFilePaths.at(inputFileIndex));

                unsigned int numElements = model.GetInputSize(i);
                if (params.m_InputTensorShapes.size() > i && params.m_InputTensorShapes[i])
                {
                    // If the user has provided a tensor shape for the current input,
                    // override numElements
                    numElements = params.m_InputTensorShapes[i]->GetNumElements();
                }

                armnnUtils::TContainer tensorData;
                PopulateTensorWithData(tensorData,
                                       numElements,
                                       params.m_InputTypes[i],
                                       qParams,
                                       dataFile);

                inputDataContainers.push_back(tensorData);
            }
            inputs.push_back(inputDataContainers);
        }

        const size_t numOutputs = inferenceModelParams.m_OutputBindings.size();

        // The user is allowed to specify the data type of each output tensor. It is used here to construct the
        // result tensors for each iteration. It is possible for the user to specify a type that does not match
        // the data type of the corresponding model output. It may not make sense, but it is historically allowed.
        // The potential problem here is a buffer overrun when a larger data type is written into the space for a
        // smaller one. Issue a warning to highlight the potential problem.
        for (unsigned int outputIdx = 0; outputIdx < model.GetOutputBindingInfos().size(); ++outputIdx)
        {
            armnn::DataType type = model.GetOutputBindingInfo(outputIdx).second.GetDataType();
            switch (type)
            {
                // --output-type only supports float, int,  qasymms8 or qasymmu8.
                case armnn::DataType::Float32:
                    if (params.m_OutputTypes[outputIdx].compare("float") != 0)
                    {
                        ARMNN_LOG(warning) << "Model output index: " << outputIdx << " has data type Float32. The " <<
                                           "corresponding --output-type is " << params.m_OutputTypes[outputIdx] <<
                                           ". This may cause unexpected problems or random failures.";
                    }
                    break;
                case armnn::DataType::QAsymmU8:
                    if (params.m_OutputTypes[outputIdx].compare("qasymmu8") != 0)
                    {
                        ARMNN_LOG(warning) << "Model output index: " << outputIdx << " has data type QAsymmU8. The " <<
                                           "corresponding --output-type is " << params.m_OutputTypes[outputIdx] <<
                                           ". This may cause unexpected problemsor random failures.";
                    }
                    break;
                case armnn::DataType::Signed32:
                    if (params.m_OutputTypes[outputIdx].compare("int") != 0)
                    {
                        ARMNN_LOG(warning) << "Model output index: " << outputIdx << " has data type Signed32. The " <<
                                           "corresponding --output-type is " << params.m_OutputTypes[outputIdx] <<
                                           ". This may cause unexpected problems or random failures.";
                    }
                    break;
                case armnn::DataType::QAsymmS8:
                    if (params.m_OutputTypes[outputIdx].compare("qasymms8") != 0)
                    {
                        ARMNN_LOG(warning) << "Model output index: " << outputIdx << " has data type QAsymmS8. The " <<
                                           "corresponding --output-type is " << params.m_OutputTypes[outputIdx] <<
                                           ". This may cause unexpected problems or random failures.";
                    }
                    break;
                default:
                    break;
            }
        }
        for (unsigned int j = 0; j < params.m_Iterations; ++j)
        {
            std::vector <armnnUtils::TContainer> outputDataContainers;
            for (unsigned int i = 0; i < numOutputs; ++i)
            {
                if (params.m_OutputTypes[i].compare("float") == 0)
                {
                    outputDataContainers.push_back(std::vector<float>(model.GetOutputSize(i)));
                }
                else if (params.m_OutputTypes[i].compare("int") == 0)
                {
                    outputDataContainers.push_back(std::vector<int>(model.GetOutputSize(i)));
                }
                else if (params.m_OutputTypes[i].compare("qasymm8") == 0 ||
                         params.m_OutputTypes[i].compare("qasymmu8") == 0)
                {
                    outputDataContainers.push_back(std::vector<uint8_t>(model.GetOutputSize(i)));
                }
                else if (params.m_OutputTypes[i].compare("qasymms8") == 0)
                {
                    outputDataContainers.push_back(std::vector<int8_t>(model.GetOutputSize(i)));
                } else
                {
                    ARMNN_LOG(fatal) << "Unsupported tensor data type \"" << params.m_OutputTypes[i] << "\". ";
                    return EXIT_FAILURE;
                }
            }
            outputs.push_back(outputDataContainers);
        }

        if (params.m_Iterations > 1)
        {
            std::stringstream msg;
            msg << "Network will be executed " << params.m_Iterations;
            if (params.m_Concurrent)
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

        // Synchronous execution
        if (!params.m_Concurrent)
        {
            for (size_t x = 0; x < params.m_Iterations; x++)
            {
                // model.Run returns the inference time elapsed in EnqueueWorkload (in milliseconds)
                auto inference_duration = model.Run(inputs[x], outputs[x]);

                if (params.m_GenerateTensorData)
                {
                    ARMNN_LOG(warning) << "The input data was generated, note that the output will not be useful";
                }
                if (params.m_DontPrintOutputs)
                {
                    ARMNN_LOG(info) << "Printing outputs to console is disabled.";
                }

                // Print output tensors
                const auto& infosOut = model.GetOutputBindingInfos();
                for (size_t i = 0; i < numOutputs; i++)
                {
                    const armnn::TensorInfo& infoOut = infosOut[i].second;

                    // We've made sure before that the number of output files either equals numOutputs, in which
                    // case we override those files when processing the results of each iteration (only the result
                    // of the last iteration will be stored), or there are enough
                    // output files for each output of each iteration.
                    size_t outputFileIndex = x * numOutputs + i;
                    if (!params.m_OutputTensorFiles.empty())
                    {
                        outputFileIndex = outputFileIndex % params.m_OutputTensorFiles.size();
                        ARMNN_LOG(info) << "Writing output " << i << " named: '"
                                        << inferenceModelParams.m_OutputBindings[i]
                                        << "' of iteration: " << x+1 << " to file: '"
                                        << params.m_OutputTensorFiles[outputFileIndex] << "'";
                    }
                    auto outputTensorFile = params.m_OutputTensorFiles.empty()
                                            ? ""
                                            : params.m_OutputTensorFiles[outputFileIndex];

                    TensorPrinter printer(inferenceModelParams.m_OutputBindings[i],
                                          infoOut,
                                          outputTensorFile,
                                          params.m_DequantizeOutput,
                                          !params.m_DontPrintOutputs);
                    mapbox::util::apply_visitor(printer, outputs[x][i]);
                }

                ARMNN_LOG(info) << "\nInference time: " << std::setprecision(2)
                                << std::fixed << inference_duration.count() << " ms\n";

                // If thresholdTime == 0.0 (default), then it hasn't been supplied at command line
                if (params.m_ThresholdTime != 0.0)
                {
                    ARMNN_LOG(info) << "Threshold time: " << std::setprecision(2)
                                    << std::fixed << params.m_ThresholdTime << " ms";
                    auto thresholdMinusInference = params.m_ThresholdTime - inference_duration.count();
                    ARMNN_LOG(info) << "Threshold time - Inference time: " << std::setprecision(2)
                                    << std::fixed << thresholdMinusInference << " ms" << "\n";

                    if (thresholdMinusInference < 0)
                    {
                        std::string errorMessage = "Elapsed inference time is greater than provided threshold time.";
                        ARMNN_LOG(fatal) << errorMessage;
                    }
                }
            }
        }
        // Asynchronous execution using the Arm NN thread pool
        else if (params.m_ThreadPoolSize >= 1)
        {
            try
            {
                ARMNN_LOG(info) << "Asynchronous execution with Arm NN thread pool...  \n";
                armnn::AsyncCallbackManager callbackManager;
                std::unordered_map<armnn::InferenceId, std::vector<armnnUtils::TContainer>&> inferenceOutputMap;

                // Declare the latest and earliest inference times here to be used when calculating overall time
                std::chrono::high_resolution_clock::time_point earliestStartTime;
                std::chrono::high_resolution_clock::time_point latestEndTime =
                    std::chrono::high_resolution_clock::now();

                // For the asynchronous execution, we are adding a pool of working memory handles (1 per thread) in the
                // LoadedNetwork with each scheduled inference having a specific priority
                for (size_t i = 0; i < params.m_Iterations; ++i)
                {
                    std::shared_ptr<armnn::AsyncExecutionCallback> cb = callbackManager.GetNewCallback();
                    inferenceOutputMap.insert({cb->GetInferenceId(), outputs[i]});
                    model.RunAsync(inputs[i], outputs[i], cb);
                }

                // Check the results
                unsigned int j = 0;
                for (size_t iteration = 0; iteration < params.m_Iterations; ++iteration)
                {
                    auto cb = callbackManager.GetNotifiedCallback();

                    // Get the results
                    auto endTime = time_point_cast<std::chrono::milliseconds>(cb->GetEndTime());
                    auto startTime = time_point_cast<std::chrono::milliseconds>(cb->GetStartTime());
                    auto inferenceDuration = endTime - startTime;

                    if (latestEndTime < cb->GetEndTime())
                    {
                        latestEndTime = cb->GetEndTime();
                    }

                    if (earliestStartTime.time_since_epoch().count() == 0)
                    {
                        earliestStartTime = cb->GetStartTime();
                    }
                    else if (earliestStartTime > cb->GetStartTime())
                    {
                        earliestStartTime = cb->GetStartTime();
                    }

                    if (params.m_GenerateTensorData)
                    {
                        ARMNN_LOG(warning) << "The input data was generated, note that the output will not be useful";
                    }
                    if (params.m_DontPrintOutputs)
                    {
                        ARMNN_LOG(info) << "Printing outputs to console is disabled.";
                    }

                    // Print output tensors
                    const auto& infosOut = model.GetOutputBindingInfos();
                    for (size_t i = 0; i < numOutputs; i++)
                    {
                        // We've made sure before that the number of output files either equals numOutputs, in which
                        // case we override those files when processing the results of each iteration (only the
                        // result of the last iteration will be stored), or there are enough
                        // output files for each output of each iteration.
                        size_t outputFileIndex = iteration * numOutputs + i;
                        if (!params.m_OutputTensorFiles.empty())
                        {
                            outputFileIndex = outputFileIndex % params.m_OutputTensorFiles.size();
                            ARMNN_LOG(info) << "Writing output " << i << " named: '"
                                            << inferenceModelParams.m_OutputBindings[i]
                                            << "' of iteration: " << iteration+1 << " to file: '"
                                            << params.m_OutputTensorFiles[outputFileIndex] << "'";
                        }

                        const armnn::TensorInfo& infoOut = infosOut[i].second;
                        auto outputTensorFile = params.m_OutputTensorFiles.empty()
                                                ? ""
                                                : params.m_OutputTensorFiles[outputFileIndex];

                        TensorPrinter printer(inferenceModelParams.m_OutputBindings[i],
                                              infoOut,
                                              outputTensorFile,
                                              params.m_DequantizeOutput,
                                              !params.m_DontPrintOutputs);
                        mapbox::util::apply_visitor(printer, inferenceOutputMap.at(cb->GetInferenceId())[i]);
                    }

                    CheckInferenceTimeThreshold(inferenceDuration, params.m_ThresholdTime);
                    ++j;
                }
                //print duration difference between overallStartTime and overallEndTime
                auto overallEndTime = time_point_cast<std::chrono::milliseconds>(latestEndTime);
                auto overallStartTime = time_point_cast<std::chrono::milliseconds>(earliestStartTime);
                auto totalInferenceDuration = overallEndTime - overallStartTime;
                ARMNN_LOG(info) << "\nOverall Inference time: " << std::setprecision(2)
                                << std::fixed << totalInferenceDuration.count() << " ms\n";
            }
            catch (const armnn::Exception& e)
            {
                ARMNN_LOG(fatal) << "Armnn Error: " << e.what();
                return EXIT_FAILURE;
            }
        }
        // Asynchronous execution using std::launch::async
        else
        {
            try
            {
                ARMNN_LOG(info) << "Asynchronous Execution with std::launch:async...  \n";
                std::vector<std::future<std::tuple<unsigned int,
                    std::chrono::duration<double, std::milli>>>> inferenceResults;
                inferenceResults.reserve(params.m_Iterations);

                // Create WorkingMemHandles for each inference
                std::vector<std::unique_ptr<armnn::experimental::IWorkingMemHandle>> workingMemHandles;
                workingMemHandles.reserve(params.m_Iterations);
                for (unsigned int i = 0; i < params.m_Iterations; ++i)
                {
                    workingMemHandles.push_back(model.CreateWorkingMemHandle());
                }

                // Run each inference in its own thread
                // start a timer
                const auto start_time = armnn::GetTimeNow();
                for (unsigned int i = 0; i < params.m_Iterations; ++i)
                {
                    armnn::experimental::IWorkingMemHandle& workingMemHandleRef = *workingMemHandles[i].get();

                    inferenceResults.push_back(std::async(
                        std::launch::async, [&model, &workingMemHandleRef, &inputs, &outputs, i]() {
                            return model.RunAsync(workingMemHandleRef, inputs[i], outputs[i], i);
                        }
                        ));
                }

                // Check the results
                for (unsigned int j = 0; j < inferenceResults.size(); ++j)
                {
                    // Get the results
                    auto inferenceResult = inferenceResults[j].get();
                    auto inferenceDuration = std::get<1>(inferenceResult);
                    auto inferenceID = std::get<0>(inferenceResult);

                    if (params.m_GenerateTensorData)
                    {
                        ARMNN_LOG(warning) << "The input data was generated, note that the output will not be useful";
                    }
                    if (params.m_DontPrintOutputs)
                    {
                        ARMNN_LOG(info) << "Printing outputs to console is disabled.";
                    }

                    // Print output tensors
                    const auto& infosOut = model.GetOutputBindingInfos();
                    for (size_t i = 0; i < numOutputs; i++)
                    {
                        // We've made sure before that the number of output files either equals numOutputs, in which
                        // case we override those files when processing the results of each iteration (only the
                        // result of the last iteration will be stored), or there are enough
                        // output files for each output of each iteration.
                        size_t outputFileIndex = j * numOutputs + i;
                        if (!params.m_OutputTensorFiles.empty())
                        {
                            outputFileIndex = outputFileIndex % params.m_OutputTensorFiles.size();
                            ARMNN_LOG(info) << "Writing output " << i << " named: '"
                                            << inferenceModelParams.m_OutputBindings[i]
                                            << "' of iteration: " << j+1 << " to file: '"
                                            << params.m_OutputTensorFiles[outputFileIndex] << "'";
                        }
                        const armnn::TensorInfo& infoOut = infosOut[i].second;
                        auto outputTensorFile = params.m_OutputTensorFiles.empty()
                                                ? ""
                                                : params.m_OutputTensorFiles[outputFileIndex];

                        TensorPrinter printer(inferenceModelParams.m_OutputBindings[i],
                                              infoOut,
                                              outputTensorFile,
                                              params.m_DequantizeOutput,
                                              !params.m_DontPrintOutputs);
                        mapbox::util::apply_visitor(printer, outputs[j][i]);
                    }
                    CheckInferenceTimeThreshold(inferenceDuration, params.m_ThresholdTime);
                    ARMNN_LOG(info) << "Asynchronous Execution is finished for Inference ID: " << inferenceID << " \n";
                }
                // finish timer
                const auto duration = armnn::GetTimeDuration(start_time);
                ARMNN_LOG(info) << "\nOverall Inference time: " << std::setprecision(2)
                                << std::fixed << duration.count() << " ms\n";
            }
            catch (const armnn::Exception& e)
            {
                ARMNN_LOG(fatal) << "Armnn Error: " << e.what();
                return EXIT_FAILURE;
            }
        }
    }
    catch (const armnn::Exception& e)
    {
        ARMNN_LOG(fatal) << "Armnn Error: " << e.what();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

// MAIN
int main(int argc, const char* argv[])
{
    // Configures logging for both the ARMNN library and this test program.
    #ifdef NDEBUG
    armnn::LogSeverity level = armnn::LogSeverity::Info;
    #else
    armnn::LogSeverity level = armnn::LogSeverity::Debug;
    #endif
    armnn::ConfigureLogging(true, true, level);


    // Get ExecuteNetwork parameters and runtime options from command line
    // This might throw an InvalidArgumentException if the user provided invalid inputs
    ProgramOptions ProgramOptions;
    try {
        ProgramOptions.ParseOptions(argc, argv);
    } catch (const std::exception &e){
        ARMNN_LOG(fatal) << e.what();
        return EXIT_FAILURE;
    }

    if ((ProgramOptions.m_ExNetParams.m_OutputDetailsToStdOut ||
         ProgramOptions.m_ExNetParams.m_OutputDetailsOnlyToStdOut)
         && !ProgramOptions.m_ExNetParams.m_EnableProfiling)
    {
        ARMNN_LOG(fatal) << "You must enable profiling if you would like to output layer details";
        return EXIT_FAILURE;
    }

    std::string modelFormat = ProgramOptions.m_ExNetParams.m_ModelFormat;

    // Forward to implementation based on the parser type
    if (modelFormat.find("armnn") != std::string::npos)
    {
    #if defined(ARMNN_SERIALIZER)
        std::shared_ptr<armnn::IRuntime> runtime(armnn::IRuntime::Create(ProgramOptions.m_RuntimeOptions));
        return MainImpl<armnnDeserializer::IDeserializer, float>(ProgramOptions.m_ExNetParams, runtime);
    #else
        ARMNN_LOG(fatal) << "Not built with serialization support.";
        return EXIT_FAILURE;
    #endif
    }
    else if (modelFormat.find("onnx") != std::string::npos)
    {
    #if defined(ARMNN_ONNX_PARSER)
        std::shared_ptr<armnn::IRuntime> runtime(armnn::IRuntime::Create(ProgramOptions.m_RuntimeOptions));
        return MainImpl<armnnOnnxParser::IOnnxParser, float>(ProgramOptions.m_ExNetParams, runtime);
    #else
        ARMNN_LOG(fatal) << "Not built with Onnx parser support.";
        return EXIT_FAILURE;
    #endif
    }
    else if(modelFormat.find("tflite") != std::string::npos)
    {
        if (ProgramOptions.m_ExNetParams.m_TfLiteExecutor == ExecuteNetworkParams::TfLiteExecutor::ArmNNTfLiteParser)
        {
            #if defined(ARMNN_TF_LITE_PARSER)
                std::shared_ptr<armnn::IRuntime> runtime(armnn::IRuntime::Create(ProgramOptions.m_RuntimeOptions));
                return MainImpl<armnnTfLiteParser::ITfLiteParser, float>(ProgramOptions.m_ExNetParams, runtime);
            #else
                ARMNN_LOG(fatal) << "Not built with Tensorflow-Lite parser support.";
                return EXIT_FAILURE;
            #endif
        }
        else if (ProgramOptions.m_ExNetParams.m_TfLiteExecutor ==
                    ExecuteNetworkParams::TfLiteExecutor::ArmNNTfLiteDelegate ||
                ProgramOptions.m_ExNetParams.m_TfLiteExecutor ==
                    ExecuteNetworkParams::TfLiteExecutor::TfliteInterpreter)
        {
        #if defined(ARMNN_TF_LITE_DELEGATE)
            return TfLiteDelegateMainImpl(ProgramOptions.m_ExNetParams, ProgramOptions.m_RuntimeOptions);
        #else
            ARMNN_LOG(fatal) << "Not built with Arm NN Tensorflow-Lite delegate support.";
            return EXIT_FAILURE;
        #endif
        }
    }
    else
    {
        ARMNN_LOG(fatal) << "Unknown model format: '" << modelFormat
                         << "'. Please include 'tflite' or 'onnx'";
        return EXIT_FAILURE;
    }
}
