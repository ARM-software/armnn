//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NetworkExecutionUtils/NetworkExecutionUtils.hpp"
#include "ExecuteNetworkProgramOptions.hpp"

#include <armnn/Logging.hpp>
#include <Filesystem.hpp>
#include <InferenceTest.hpp>

#if defined(ARMNN_SERIALIZER)
#include "armnnDeserializer/IDeserializer.hpp"
#endif
#if defined(ARMNN_CAFFE_PARSER)
#include "armnnCaffeParser/ICaffeParser.hpp"
#endif
#if defined(ARMNN_TF_PARSER)
#include "armnnTfParser/ITfParser.hpp"
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
#if defined(ARMNN_TFLITE_DELEGATE)
int TfLiteDelegateMainImpl(const ExecuteNetworkParams& params,
                           const std::shared_ptr<armnn::IRuntime>& runtime = nullptr)
{
    using namespace tflite;

    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(params.m_ModelPath.c_str());

    auto tfLiteInterpreter =  std::make_unique<Interpreter>();
    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&tfLiteInterpreter);
    tfLiteInterpreter->AllocateTensors();

    // Create the Armnn Delegate
    armnnDelegate::DelegateOptions delegateOptions(params.m_ComputeDevices);
    std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
            theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                             armnnDelegate::TfLiteArmnnDelegateDelete);
    // Register armnn_delegate to TfLiteInterpreter
    int status = tfLiteInterpreter->ModifyGraphWithDelegate(std::move(theArmnnDelegate));

    std::vector<std::string>  inputBindings;
    for (const std::string& inputName: params.m_InputNames)
    {
        inputBindings.push_back(inputName);
    }

    armnn::Optional<std::string> dataFile = params.m_GenerateTensorData
                                            ? armnn::EmptyOptional()
                                            : armnn::MakeOptional<std::string>(params.m_InputTensorDataFilePaths[0]);

    const size_t numInputs = inputBindings.size();

    for(unsigned int inputIndex = 0; inputIndex < numInputs; ++inputIndex)
    {
        int input = tfLiteInterpreter->inputs()[inputIndex];
        TfLiteIntArray* inputDims = tfLiteInterpreter->tensor(input)->dims;

        long inputSize = 1;
        for (unsigned int dim = 0; dim < static_cast<unsigned int>(inputDims->size); ++dim)
        {
            inputSize *=  inputDims->data[dim];
        }

        if (params.m_InputTypes[inputIndex].compare("float") == 0)
        {
            auto inputData = tfLiteInterpreter->typed_tensor<float>(input);
            std::vector<float> tensorData;
            PopulateTensorWithDataGeneric<float>(tensorData,
                                                  params.m_InputTensorShapes[inputIndex]->GetNumElements(),
                                                  dataFile,
                                                  [](const std::string& s)
                                                  { return std::stof(s); });

            std::copy(tensorData.begin(), tensorData.end(), inputData);
        }
        else if (params.m_InputTypes[inputIndex].compare("int8") == 0)
        {
            auto inputData = tfLiteInterpreter->typed_tensor<int8_t>(input);
            std::vector<int8_t> tensorData;
            PopulateTensorWithDataGeneric<int8_t>(tensorData,
                                                  params.m_InputTensorShapes[inputIndex]->GetNumElements(),
                                                  dataFile,
                                                  [](const std::string& s)
                                                  { return armnn::numeric_cast<int8_t>(std::stoi(s)); });

            std::copy(tensorData.begin(), tensorData.end(), inputData);
        }
        else if (params.m_InputTypes[inputIndex].compare("int") == 0)
        {
            auto inputData = tfLiteInterpreter->typed_tensor<int32_t>(input);
            std::vector<int32_t> tensorData;
            PopulateTensorWithDataGeneric<int32_t>(tensorData,
                                                   params.m_InputTensorShapes[inputIndex]->GetNumElements(),
                                                   dataFile,
                                                   [](const std::string& s)
                                                   { return std::stoi(s); });

            std::copy(tensorData.begin(), tensorData.end(), inputData);
        }
        else if (params.m_InputTypes[inputIndex].compare("qasymm8") == 0)
        {
            auto inputData = tfLiteInterpreter->typed_tensor<uint8_t>(input);
            std::vector<uint8_t> tensorData;
            PopulateTensorWithDataGeneric<uint8_t>(tensorData,
                                                   params.m_InputTensorShapes[inputIndex]->GetNumElements(),
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

    for (size_t x = 0; x < params.m_Iterations; x++)
    {
        // Run the inference
        tfLiteInterpreter->Invoke();

        // Print out the output
        for (unsigned int outputIndex = 0; outputIndex < params.m_OutputNames.size(); ++outputIndex)
        {
            auto tfLiteDelegateOutputId = tfLiteInterpreter->outputs()[outputIndex];
            TfLiteIntArray* outputDims = tfLiteInterpreter->tensor(tfLiteDelegateOutputId)->dims;

            long outputSize = 1;
            for (unsigned int dim = 0; dim < static_cast<unsigned int>(outputDims->size); ++dim)
            {
                outputSize *=  outputDims->data[dim];
            }

            std::cout << params.m_OutputNames[outputIndex] << ": ";
            if (params.m_OutputTypes[outputIndex].compare("float") == 0)
            {
                auto tfLiteDelageOutputData = tfLiteInterpreter->typed_tensor<float>(tfLiteDelegateOutputId);
                if(tfLiteDelageOutputData == NULL)
                {
                    ARMNN_LOG(fatal) << "Output tensor is null, output type: "
                                        "\"" << params.m_OutputTypes[outputIndex] << "\" may be incorrect.";
                    return EXIT_FAILURE;
                }

                for (int i = 0; i < outputSize; ++i)
                {
                    std::cout << tfLiteDelageOutputData[i] << ", ";
                    if (i % 60 == 0)
                    {
                        std::cout << std::endl;
                    }
                }
            }
            else if (params.m_OutputTypes[outputIndex].compare("int") == 0)
            {
                auto tfLiteDelageOutputData = tfLiteInterpreter->typed_tensor<int32_t>(tfLiteDelegateOutputId);
                if(tfLiteDelageOutputData == NULL)
                {
                    ARMNN_LOG(fatal) << "Output tensor is null, output type: "
                                        "\"" << params.m_OutputTypes[outputIndex] << "\" may be incorrect.";
                    return EXIT_FAILURE;
                }

                for (int i = 0; i < outputSize; ++i)
                {
                    std::cout << tfLiteDelageOutputData[i] << ", ";
                    if (i % 60 == 0)
                    {
                        std::cout << std::endl;
                    }
                }
            }
            else if (params.m_OutputTypes[outputIndex].compare("int8") == 0)
            {
                auto tfLiteDelageOutputData = tfLiteInterpreter->typed_tensor<int8_t>(tfLiteDelegateOutputId);
                if(tfLiteDelageOutputData == NULL)
                {
                    ARMNN_LOG(fatal) << "Output tensor is null, output type: "
                                        "\"" << params.m_OutputTypes[outputIndex] << "\" may be incorrect.";
                    return EXIT_FAILURE;
                }

                for (int i = 0; i < outputSize; ++i)
                {
                    std::cout << signed(tfLiteDelageOutputData[i]) << ", ";
                    if (i % 60 == 0)
                    {
                        std::cout << std::endl;
                    }
                }
            }
            else if (params.m_OutputTypes[outputIndex].compare("qasymm8") == 0)
            {
                auto tfLiteDelageOutputData = tfLiteInterpreter->typed_tensor<uint8_t>(tfLiteDelegateOutputId);
                if(tfLiteDelageOutputData == NULL)
                {
                    ARMNN_LOG(fatal) << "Output tensor is null, output type: "
                                        "\"" << params.m_OutputTypes[outputIndex] << "\" may be incorrect.";
                    return EXIT_FAILURE;
                }

                for (int i = 0; i < outputSize; ++i)
                {
                    std::cout << unsigned(tfLiteDelageOutputData[i]) << ", ";
                    if (i % 60 == 0)
                    {
                        std::cout << std::endl;
                    }
                }
            }
            else
            {
                ARMNN_LOG(fatal) << "Output tensor is null, output type: "
                                    "\"" << params.m_OutputTypes[outputIndex] <<
                                 "\" may be incorrect. Output type can be specified with -z argument";
                return EXIT_FAILURE;
            }
            std::cout << std::endl;
        }
    }

    return status;
}
#endif
template<typename TParser, typename TDataType>
int MainImpl(const ExecuteNetworkParams& params,
             const std::shared_ptr<armnn::IRuntime>& runtime = nullptr)
{
    using TContainer = mapbox::util::variant<std::vector<float>, std::vector<int>, std::vector<unsigned char>>;

    std::vector<TContainer> inputDataContainers;

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
        for(unsigned int i = 0; i < numInputs; ++i)
        {
            armnn::Optional<QuantizationParams> qParams = params.m_QuantizeInput ?
                                                          armnn::MakeOptional<QuantizationParams>(
                                                                  model.GetInputQuantizationParams()) :
                                                          armnn::EmptyOptional();

            armnn::Optional<std::string> dataFile = params.m_GenerateTensorData ?
                                                    armnn::EmptyOptional() :
                                                    armnn::MakeOptional<std::string>(
                                                            params.m_InputTensorDataFilePaths[i]);

            unsigned int numElements = model.GetInputSize(i);
            if (params.m_InputTensorShapes.size() > i && params.m_InputTensorShapes[i])
            {
                // If the user has provided a tensor shape for the current input,
                // override numElements
                numElements = params.m_InputTensorShapes[i]->GetNumElements();
            }

            TContainer tensorData;
            PopulateTensorWithData(tensorData,
                                   numElements,
                                   params.m_InputTypes[i],
                                   qParams,
                                   dataFile);

            inputDataContainers.push_back(tensorData);
        }

        const size_t numOutputs = inferenceModelParams.m_OutputBindings.size();
        std::vector<TContainer> outputDataContainers;

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
            else if (params.m_OutputTypes[i].compare("qasymm8") == 0)
            {
                outputDataContainers.push_back(std::vector<uint8_t>(model.GetOutputSize(i)));
            }
            else
            {
                ARMNN_LOG(fatal) << "Unsupported tensor data type \"" << params.m_OutputTypes[i] << "\". ";
                return EXIT_FAILURE;
            }
        }

        for (size_t x = 0; x < params.m_Iterations; x++)
        {
            // model.Run returns the inference time elapsed in EnqueueWorkload (in milliseconds)
            auto inference_duration = model.Run(inputDataContainers, outputDataContainers);

            if (params.m_GenerateTensorData)
            {
                ARMNN_LOG(warning) << "The input data was generated, note that the output will not be useful";
            }

            // Print output tensors
            const auto& infosOut = model.GetOutputBindingInfos();
            for (size_t i = 0; i < numOutputs; i++)
            {
                const armnn::TensorInfo& infoOut = infosOut[i].second;
                auto outputTensorFile = params.m_OutputTensorFiles.empty() ? "" : params.m_OutputTensorFiles[i];

                TensorPrinter printer(inferenceModelParams.m_OutputBindings[i],
                                      infoOut,
                                      outputTensorFile,
                                      params.m_DequantizeOutput);
                mapbox::util::apply_visitor(printer, outputDataContainers[i]);
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
    ProgramOptions ProgramOptions(argc, argv);

    // Create runtime
    std::shared_ptr<armnn::IRuntime> runtime(armnn::IRuntime::Create(ProgramOptions.m_RuntimeOptions));

    std::string modelFormat = ProgramOptions.m_ExNetParams.m_ModelFormat;

    // Forward to implementation based on the parser type
    if (modelFormat.find("armnn") != std::string::npos)
    {
    #if defined(ARMNN_SERIALIZER)
        return MainImpl<armnnDeserializer::IDeserializer, float>(ProgramOptions.m_ExNetParams, runtime);
    #else
        ARMNN_LOG(fatal) << "Not built with serialization support.";
        return EXIT_FAILURE;
    #endif
    }
    else if (modelFormat.find("caffe") != std::string::npos)
    {
    #if defined(ARMNN_CAFFE_PARSER)
        return MainImpl<armnnCaffeParser::ICaffeParser, float>(ProgramOptions.m_ExNetParams, runtime);
    #else
        ARMNN_LOG(fatal) << "Not built with Caffe parser support.";
        return EXIT_FAILURE;
    #endif
    }
    else if (modelFormat.find("onnx") != std::string::npos)
    {
    #if defined(ARMNN_ONNX_PARSER)
        return MainImpl<armnnOnnxParser::IOnnxParser, float>(ProgramOptions.m_ExNetParams, runtime);
    #else
        ARMNN_LOG(fatal) << "Not built with Onnx parser support.";
        return EXIT_FAILURE;
    #endif
    }
    else if (modelFormat.find("tensorflow") != std::string::npos)
    {
    #if defined(ARMNN_TF_PARSER)
        return MainImpl<armnnTfParser::ITfParser, float>(ProgramOptions.m_ExNetParams, runtime);
    #else
        ARMNN_LOG(fatal) << "Not built with Tensorflow parser support.";
        return EXIT_FAILURE;
    #endif
    }
    else if(modelFormat.find("tflite") != std::string::npos)
    {

        if (ProgramOptions.m_ExNetParams.m_EnableDelegate)
        {
        #if defined(ARMNN_TF_LITE_DELEGATE)
            return TfLiteDelegateMainImpl(ProgramOptions.m_ExNetParams, runtime);
        #else
            ARMNN_LOG(fatal) << "Not built with Tensorflow-Lite parser support.";
            return EXIT_FAILURE;
        #endif
        }
    #if defined(ARMNN_TF_LITE_PARSER)
        return MainImpl<armnnTfLiteParser::ITfLiteParser, float>(ProgramOptions.m_ExNetParams, runtime);
    #else
        ARMNN_LOG(fatal) << "Not built with Tensorflow-Lite parser support.";
        return EXIT_FAILURE;
    #endif
    }
    else
    {
        ARMNN_LOG(fatal) << "Unknown model format: '" << modelFormat
                         << "'. Please include 'caffe', 'tensorflow', 'tflite' or 'onnx'";
        return EXIT_FAILURE;
    }
}
