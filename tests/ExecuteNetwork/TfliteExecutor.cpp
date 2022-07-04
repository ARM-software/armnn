//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TfliteExecutor.hpp"

TfLiteExecutor::TfLiteExecutor(const ExecuteNetworkParams& params) : m_Params(params)
{
    std::unique_ptr<tflite::FlatBufferModel> model =
            tflite::FlatBufferModel::BuildFromFile(m_Params.m_ModelPath.c_str());

    m_TfLiteInterpreter =  std::make_unique<Interpreter>();
    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder builder(*model, resolver);
    builder(&m_TfLiteInterpreter);
    m_TfLiteInterpreter->AllocateTensors();

    int status;
    if (m_Params.m_TfLiteExecutor == ExecuteNetworkParams::TfLiteExecutor::ArmNNTfLiteDelegate)
    {
        // Create the Armnn Delegate
        // Populate a DelegateOptions from the ExecuteNetworkParams.
        armnnDelegate::DelegateOptions delegateOptions = m_Params.ToDelegateOptions();
        delegateOptions.SetExternalProfilingParams(delegateOptions.GetExternalProfilingParams());

        std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
                theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                                 armnnDelegate::TfLiteArmnnDelegateDelete);
        // Register armnn_delegate to TfLiteInterpreter
        status = m_TfLiteInterpreter->ModifyGraphWithDelegate(std::move(theArmnnDelegate));
        if (status == kTfLiteError)
        {
            LogAndThrow("Could not register ArmNN TfLite Delegate to TfLiteInterpreter");
        }
    }
    else
    {
        std::cout << "Running on TfLite without ArmNN delegate\n";
    }

    armnn::Optional<std::string> dataFile = m_Params.m_GenerateTensorData
                                            ? armnn::EmptyOptional()
                                            : armnn::MakeOptional<std::string>(m_Params.m_InputTensorDataFilePaths[0]);

    const size_t numInputs = m_Params.m_InputNames.size();

    for(unsigned int inputIndex = 0; inputIndex < numInputs; ++inputIndex)
    {
        int input = m_TfLiteInterpreter->inputs()[inputIndex];

        TfLiteIntArray* inputDims = m_TfLiteInterpreter->tensor(input)->dims;

        unsigned int inputSize = 1;
        for (unsigned int dim = 0; dim < static_cast<unsigned int>(inputDims->size); ++dim)
        {
            inputSize *= inputDims->data[dim];
        }

        const auto& inputName = m_TfLiteInterpreter->input_tensor(input)->name;
        const auto& dataType = m_TfLiteInterpreter->input_tensor(input)->type;

        switch (dataType)
        {
            case kTfLiteFloat32:
            {
                auto inputData = m_TfLiteInterpreter->typed_tensor<float>(input);
                PopulateTensorWithData(inputData, inputSize, dataFile, inputName);
                break;
            }
            case kTfLiteInt32:
            {
                auto inputData = m_TfLiteInterpreter->typed_tensor<int>(input);
                PopulateTensorWithData(inputData, inputSize, dataFile, inputName);
                break;
            }
            case kTfLiteUInt8:
            {
                auto inputData = m_TfLiteInterpreter->typed_tensor<uint8_t>(input);
                PopulateTensorWithData(inputData, inputSize, dataFile, inputName);
                break;
            }
            case kTfLiteInt16:
            {
                auto inputData = m_TfLiteInterpreter->typed_tensor<int16_t>(input);
                PopulateTensorWithData(inputData, inputSize, dataFile, inputName);
                break;
            }
            case kTfLiteInt8:
            {
                auto inputData = m_TfLiteInterpreter->typed_tensor<int8_t>(input);
                PopulateTensorWithData(inputData, inputSize, dataFile, inputName);
                break;
            }
            default:
            {
                LogAndThrow("Unsupported input tensor data type");
            }
        }
    }
}

std::vector<const void *> TfLiteExecutor::Execute()
{
    int status = 0;
    std::vector<const void*> results;
    for (size_t x = 0; x < m_Params.m_Iterations; x++)
    {
        // Start timer to record inference time in milliseconds.
        const auto start_time = armnn::GetTimeNow();
        // Run the inference
        status = m_TfLiteInterpreter->Invoke();
        const auto duration = armnn::GetTimeDuration(start_time);

        if (m_Params.m_DontPrintOutputs || m_Params.m_ReuseBuffers)
        {
            break;
        }
        // Print out the output
        for (unsigned int outputIndex = 0; outputIndex < m_TfLiteInterpreter->outputs().size(); ++outputIndex)
        {
            auto tfLiteDelegateOutputId = m_TfLiteInterpreter->outputs()[outputIndex];
            TfLiteIntArray* outputDims = m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->dims;
            // If we've been asked to write to a file then set a file output stream. Otherwise use stdout.
            FILE* outputTensorFile = stdout;
            if (!m_Params.m_OutputTensorFiles.empty())
            {
                outputTensorFile = fopen(m_Params.m_OutputTensorFiles[outputIndex].c_str(), "w");
                if (outputTensorFile == NULL)
                {
                    LogAndThrow("Specified output tensor file, \"" + m_Params.m_OutputTensorFiles[outputIndex] +
                                "\", cannot be created. Defaulting to stdout. Error was: " + std::strerror(errno));
                }
                else
                {
                    ARMNN_LOG(info) << "Writing output " << outputIndex << "' of iteration: " << x+1 << " to file: '"
                                    << m_Params.m_OutputTensorFiles[outputIndex] << "'";
                }
            }
            long outputSize = 1;
            for (unsigned int dim = 0; dim < static_cast<unsigned int>(outputDims->size); ++dim)
            {
                outputSize *=  outputDims->data[dim];
            }

            std::cout << m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->name << ": ";
            results.push_back(m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->allocation);

            switch (m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->type)
            {

                case kTfLiteFloat32:
                {
                    auto tfLiteDelageOutputData = m_TfLiteInterpreter->typed_tensor<float>(tfLiteDelegateOutputId);

                    for (int i = 0; i < outputSize; ++i)
                    {
                        fprintf(outputTensorFile, "%f ", tfLiteDelageOutputData[i]);
                    }
                    break;
                }
                case kTfLiteInt32:
                {
                    auto tfLiteDelageOutputData = m_TfLiteInterpreter->typed_tensor<int32_t>(tfLiteDelegateOutputId);
                    for (int i = 0; i < outputSize; ++i)
                    {
                        fprintf(outputTensorFile, "%d ", tfLiteDelageOutputData[i]);
                    }
                    break;
                }
                case kTfLiteUInt8:
                {
                    auto tfLiteDelageOutputData = m_TfLiteInterpreter->typed_tensor<uint8_t>(tfLiteDelegateOutputId);
                    for (int i = 0; i < outputSize; ++i)
                    {
                        fprintf(outputTensorFile, "%u ", tfLiteDelageOutputData[i]);
                    }
                    break;
                }
                case kTfLiteInt8:
                {
                    auto tfLiteDelageOutputData = m_TfLiteInterpreter->typed_tensor<int8_t>(tfLiteDelegateOutputId);
                    for (int i = 0; i < outputSize; ++i)
                    {
                        fprintf(outputTensorFile, "%d ", tfLiteDelageOutputData[i]);
                    }
                    break;
                }
                default:
                {
                    LogAndThrow("Unsupported output type");
                }
            }

            std::cout << std::endl;
        }
        CheckInferenceTimeThreshold(duration, m_Params.m_ThresholdTime);
    }

    std::cout << status;
    return results;
}

void TfLiteExecutor::CompareAndPrintResult(std::vector<const void*> otherOutput)
{
    for (unsigned int outputIndex = 0; outputIndex < m_TfLiteInterpreter->outputs().size(); ++outputIndex)
    {
        auto tfLiteDelegateOutputId = m_TfLiteInterpreter->outputs()[outputIndex];
        float result = 0;
        switch (m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->type)
        {
            case kTfLiteFloat32:
            {
                result =  ComputeRMSE<float>(m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->allocation,
                                                  otherOutput[outputIndex],
                                                  m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->bytes);

                break;
            }
            case kTfLiteInt32:
            {
                result =  ComputeRMSE<int32_t>(m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->allocation,
                                                    otherOutput[outputIndex],
                                                    m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->bytes);
                break;
            }
            case kTfLiteUInt8:
            {
                result =  ComputeRMSE<uint8_t>(m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->allocation,
                                                    otherOutput[outputIndex],
                                                    m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->bytes);
                break;
            }
            case kTfLiteInt8:
            {
                result =  ComputeRMSE<int8_t>(m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->allocation,
                                                    otherOutput[outputIndex],
                                                    m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->bytes);
                break;
            }
            default:
            {
            }
        }

        std::cout << "RMSE of "
                  << m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->name
                  << ": " << result << std::endl;
    }
};
