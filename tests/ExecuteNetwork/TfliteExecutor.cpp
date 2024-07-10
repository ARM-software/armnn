//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#if defined(ARMNN_TFLITE_OPAQUE_DELEGATE)
#include <../delegate/opaque/include/armnn_delegate.hpp>
#endif

#include <tensorflow/lite/core/c/c_api.h>
#include "TfliteExecutor.hpp"
#include <tensorflow/lite/kernels/kernel_util.h>

#include <../delegate/common/src/DelegateUtils.hpp>

#include <chrono>
#include <string>
#include <thread>

std::string TfLiteStatusToString(const TfLiteStatus status)
{
    switch (status)
    {
        case kTfLiteOk:
            return "Status: Ok.";
        // Generally referring to an error in the runtime (i.e. interpreter)
        case kTfLiteError:
            return "Status: Tf runtime error.";
        // Generally referring to an error from a TfLiteDelegate itself.
        case kTfLiteDelegateError:
            return "Status: The loaded delegate has returned an error.";
        // Generally referring to an error in applying a delegate due to
        // incompatibility between runtime and delegate, e.g., this error is returned
        // when trying to apply a TF Lite delegate onto a model graph that's already
        // immutable.
        case kTfLiteApplicationError:
            return "Status: Application error. An incompatibility between the Tf runtime and the loaded delegate.";
        // Generally referring to serialized delegate data not being found.
        // See tflite::delegates::Serialization.
        case kTfLiteDelegateDataNotFound:
            return "Status: data not found.";
        // Generally referring to data-writing issues in delegate serialization.
        // See tflite::delegates::Serialization.
        case kTfLiteDelegateDataWriteError:
            return "Status: Error writing serialization data.";
        // Generally referring to data-reading issues in delegate serialization.
        // See tflite::delegates::Serialization.
        case kTfLiteDelegateDataReadError:
            return "Status: Error reading serialization data.";
        // Generally referring to issues when the TF Lite model has ops that cannot be
        // resolved at runtime. This could happen when the specific op is not
        // registered or built with the TF Lite framework.
        case kTfLiteUnresolvedOps:
            return "Status: Model contains an operation that is not recognised by the runtime.";
        // Generally referring to invocation cancelled by the user.
        case kTfLiteCancelled:
            return "Status: invocation has been cancelled by the user.";
    }
    return "Unknown status result.";
}

TfLiteExecutor::TfLiteExecutor(const ExecuteNetworkParams& params, armnn::IRuntime::CreationOptions runtimeOptions)
                             : m_Params(params)
{
    using namespace std::chrono_literals;
    m_Model = tflite::FlatBufferModel::BuildFromFile(m_Params.m_ModelPath.c_str());
    if (!m_Model)
    {
        LogAndThrow("Failed to load TfLite model from: " + m_Params.m_ModelPath);
    }
    m_TfLiteInterpreter =  std::make_unique<Interpreter>();
    tflite::ops::builtin::BuiltinOpResolver resolver;

    tflite::InterpreterBuilder builder(*m_Model, resolver);

    if (m_Params.m_TfLiteExecutor == ExecuteNetworkParams::TfLiteExecutor::ArmNNTfLiteOpaqueDelegate)
    {
#if defined(ARMNN_TFLITE_OPAQUE_DELEGATE)
        if (builder(&m_TfLiteInterpreter) != kTfLiteOk)
        {
            LogAndThrow("Error loading the model into the TfLiteInterpreter.");
        }
        // Populate a DelegateOptions from the ExecuteNetworkParams.
        armnnDelegate::DelegateOptions delegateOptions = m_Params.ToDelegateOptions();
        delegateOptions.SetRuntimeOptions(runtimeOptions);
        std::unique_ptr<TfLiteDelegate, decltype(&armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateDelete)>
                theArmnnDelegate(armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateCreate(delegateOptions),
                                 armnnOpaqueDelegate::TfLiteArmnnOpaqueDelegateDelete);

        // Register armnn_delegate to TfLiteInterpreter
        auto result = m_TfLiteInterpreter->ModifyGraphWithDelegate(std::move(theArmnnDelegate));
        if (result != kTfLiteOk)
        {
            LogAndThrow("Could not register ArmNN TfLite Opaque Delegate to TfLiteInterpreter: " +
                        TfLiteStatusToString(result) + ".");
        }
#else
        LogAndThrow("Not built with Arm NN Tensorflow-Lite opaque delegate support.");
#endif
    }
    else if (m_Params.m_TfLiteExecutor == ExecuteNetworkParams::TfLiteExecutor::ArmNNTfLiteDelegate)
    {
#if defined(ARMNN_TFLITE_DELEGATE)
        if (builder(&m_TfLiteInterpreter) != kTfLiteOk)
        {
            LogAndThrow("Error loading the model into the TfLiteInterpreter.");
        }
        // Create the Armnn Delegate
        // Populate a DelegateOptions from the ExecuteNetworkParams.
        armnnDelegate::DelegateOptions delegateOptions = m_Params.ToDelegateOptions();
        delegateOptions.SetRuntimeOptions(runtimeOptions);
        std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)>
                theArmnnDelegate(armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions),
                                 armnnDelegate::TfLiteArmnnDelegateDelete);
        // Register armnn_delegate to TfLiteInterpreter
        auto result = m_TfLiteInterpreter->ModifyGraphWithDelegate(std::move(theArmnnDelegate));
        if (result != kTfLiteOk)
        {
            // We'll make an exception for kTfLiteApplicationError and allow it through in special circumstances.
            if (result == kTfLiteApplicationError)
            {
                std::cout << std::endl;
                ARMNN_LOG(warning) << "*****";
                ARMNN_LOG(warning) << "***** Calling ModifyGraphWithDelegate on the TfLite runtime resulted in "
                                      "kTfLiteApplicationError. We will allow inference to continue but be warned the "
                                      "results may be surprising!";
                ARMNN_LOG(warning) << "***** There will now be a 5 second delay.";
                ARMNN_LOG(warning) << "*****\n";
                // Insert an intentional delay so the user is aware of this significant warning.
                std::this_thread::sleep_for(5000ms);
            }
            else
            {
                LogAndThrow("Could not register ArmNN TfLite Delegate to TfLiteInterpreter: " +
                            TfLiteStatusToString(result) + ".");
            }
        }
#else
        LogAndThrow("Not built with Arm NN Tensorflow-Lite delegate support.");
#endif
    }
    else
    {
        if (builder(&m_TfLiteInterpreter) != kTfLiteOk)
        {
            LogAndThrow("Error loading the model into the TfLiteInterpreter.");
        }
        std::cout << "Running on TfLite without ArmNN delegate\n";
    }

    if (m_TfLiteInterpreter->AllocateTensors() != kTfLiteOk)
    {
        LogAndThrow("Failed to allocate tensors in the TfLiteInterpreter.");
    }

    const size_t numInputs = m_TfLiteInterpreter->inputs().size();

    for(unsigned int inputIndex = 0; inputIndex < numInputs; ++inputIndex)
    {
        armnn::Optional<std::string> dataFile = m_Params.m_GenerateTensorData
            ? armnn::EmptyOptional()
            : armnn::MakeOptional<std::string>(m_Params.m_InputTensorDataFilePaths[inputIndex]);

        int input = m_TfLiteInterpreter->inputs()[inputIndex];
        const auto& inputName = m_TfLiteInterpreter->tensor(input)->name;

        // Before we start, check if the tensor is constant.
        if (!tflite::IsConstantTensor(m_TfLiteInterpreter->tensor(input)))
        {
            TfLiteIntArray* inputDims = m_TfLiteInterpreter->tensor(input)->dims;

            unsigned int inputSize = 1;
            for (unsigned int dim = 0; dim < static_cast<unsigned int>(inputDims->size); ++dim)
            {
                inputSize *= inputDims->data[dim];
            }

            const auto& dataType = m_TfLiteInterpreter->tensor(input)->type;

            switch (dataType)
            {
                case kTfLiteFloat32:
                {
                    auto inputData = m_TfLiteInterpreter->typed_tensor<float>(input);
                    PopulateTensorWithData<float>(inputData, inputSize, dataFile, inputName);
                    break;
                }
                case kTfLiteInt32:
                {
                    auto inputData = m_TfLiteInterpreter->typed_tensor<int32_t>(input);
                    PopulateTensorWithData<int32_t>(inputData, inputSize, dataFile, inputName);
                    break;
                }
                case kTfLiteUInt8:
                {
                    auto inputData = m_TfLiteInterpreter->typed_tensor<uint8_t>(input);
                    PopulateTensorWithData<uint8_t>(inputData, inputSize, dataFile, inputName);
                    break;
                }
                case kTfLiteInt16:
                {
                    auto inputData = m_TfLiteInterpreter->typed_tensor<int16_t>(input);
                    PopulateTensorWithData<int16_t>(inputData, inputSize, dataFile, inputName);
                    break;
                }
                case kTfLiteInt8:
                {
                    auto inputData = m_TfLiteInterpreter->typed_tensor<int8_t>(input);
                    PopulateTensorWithData<int8_t>(inputData, inputSize, dataFile, inputName);
                    break;
                }
                default:
                {
                    LogAndThrow("Unsupported input tensor data type");
                }
            }
        }
        else
        {
            ARMNN_LOG(info) << "Input tensor \"" << inputName << "\" is constant and will not be populated with data.";
        }
    }
}

std::vector<const void *> TfLiteExecutor::Execute()
{
    TfLiteStatus status;
    std::vector<const void*> results;
    for (size_t x = 0; x < m_Params.m_Iterations; x++)
    {
        // Start timer to record inference time in milliseconds.
        const auto start_time = armnn::GetTimeNow();
        // Run the inference
        status = m_TfLiteInterpreter->Invoke();
        if (status != kTfLiteOk)
        {
            LogAndThrow("Failed to execute the inference on the TfLite runtime.. The result was: " +
                        TfLiteStatusToString(status) + ".");
        }
        const auto duration = armnn::GetTimeDuration(start_time);

        if (!m_Params.m_DontPrintOutputs)
        {
            // Print out the output
            for (unsigned int outputIndex = 0; outputIndex < m_TfLiteInterpreter->outputs().size(); ++outputIndex)
            {
                auto tfLiteDelegateOutputId = m_TfLiteInterpreter->outputs()[outputIndex];
                TfLiteIntArray* outputDims = m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->dims;
                // If we've been asked to write to a file then set a file output stream. Otherwise use stdout.
                FILE* outputTensorFile = stdout;
                bool isNumpyOutput = false;
                if (!m_Params.m_OutputTensorFiles.empty())
                {
                    isNumpyOutput = m_Params.m_OutputTensorFiles[outputIndex].find(".npy") != std::string::npos;
                    outputTensorFile = fopen(m_Params.m_OutputTensorFiles[outputIndex].c_str(), "w");
                    if (outputTensorFile == NULL)
                    {
                        LogAndThrow("Specified output tensor file, \"" + m_Params.m_OutputTensorFiles[outputIndex] +
                                    "\", cannot be created. Defaulting to stdout. Error was: " + std::strerror(errno));
                    }
                    else
                    {
                        ARMNN_LOG(info) << "Writing output " << outputIndex << " of iteration: " << x + 1
                                        << " to file: '" << m_Params.m_OutputTensorFiles[outputIndex] << "'";
                    }
                }

                long outputSize = 1;
                for (unsigned int dim = 0; dim < static_cast<unsigned int>(outputDims->size); ++dim)
                {
                    outputSize *= outputDims->data[dim];
                }

                // outputDims->data can be a Flexible Array Member (int data[];) in a C extern code in TF common.h
                // TensorShape constructor argument is an unsigned int *
                // so reinterpret_cast is used here to ensure the correct type of data is passed
                armnn::TensorShape shape(static_cast<unsigned int>(outputDims->size),
                                            reinterpret_cast<unsigned int *>(outputDims->data));
                armnn::DataType dataType(GetDataType(*m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)));

                std::cout << m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->name << ": ";
                switch (m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->type)
                {
                    case kTfLiteFloat32:
                    {
                        auto tfLiteDelegateOutputData = m_TfLiteInterpreter->typed_tensor<float>(
                                tfLiteDelegateOutputId);
                        results.push_back(tfLiteDelegateOutputData);

                        if (isNumpyOutput)
                        {
                            armnnNumpy::WriteToNumpyFile(m_Params.m_OutputTensorFiles[outputIndex],
                                                         tfLiteDelegateOutputData,
                                                         outputSize,
                                                         dataType,
                                                         shape);
                        }
                        else
                        {
                            for (int i = 0; i < outputSize; ++i)
                            {
                                fprintf(outputTensorFile, "%f ", tfLiteDelegateOutputData[i]);
                            }
                        }
                        break;
                    }
                    case kTfLiteInt32:
                    {
                        auto tfLiteDelegateOutputData = m_TfLiteInterpreter->typed_tensor<int32_t>(
                                tfLiteDelegateOutputId);
                        results.push_back(tfLiteDelegateOutputData);

                        if (isNumpyOutput)
                        {
                            armnnNumpy::WriteToNumpyFile(m_Params.m_OutputTensorFiles[outputIndex],
                                                         tfLiteDelegateOutputData,
                                                         outputSize,
                                                         dataType,
                                                         shape);
                        }
                        else
                        {
                            for (int i = 0; i < outputSize; ++i)
                            {
                                fprintf(outputTensorFile, "%d ", tfLiteDelegateOutputData[i]);
                            }
                        }

                        break;
                    }
                    case kTfLiteUInt8:
                    {
                        auto tfLiteDelegateOutputData = m_TfLiteInterpreter->typed_tensor<uint8_t>(
                                tfLiteDelegateOutputId);
                        results.push_back(tfLiteDelegateOutputData);

                        if (isNumpyOutput)
                        {
                            armnnNumpy::WriteToNumpyFile(m_Params.m_OutputTensorFiles[outputIndex],
                                                         tfLiteDelegateOutputData,
                                                         outputSize,
                                                         dataType,
                                                         shape);
                        }
                        else
                        {
                            for (int i = 0; i < outputSize; ++i)
                            {
                                fprintf(outputTensorFile, "%u ", tfLiteDelegateOutputData[i]);
                            }
                        }

                        break;
                    }
                    case kTfLiteInt8:
                    {
                        auto tfLiteDelegateOutputData = m_TfLiteInterpreter->typed_tensor<int8_t>(
                                tfLiteDelegateOutputId);
                        results.push_back(tfLiteDelegateOutputData);

                        if (isNumpyOutput)
                        {
                            armnnNumpy::WriteToNumpyFile(m_Params.m_OutputTensorFiles[outputIndex],
                                                         tfLiteDelegateOutputData,
                                                         outputSize,
                                                         dataType,
                                                         shape);
                        }
                        else
                        {
                            for (int i = 0; i < outputSize; ++i)
                            {
                                fprintf(outputTensorFile, "%d ", tfLiteDelegateOutputData[i]);
                            }
                        }

                        break;
                    }
                    case kTfLiteBool:
                    {
                        auto tfLiteDelegateOutputData = m_TfLiteInterpreter->typed_tensor<bool>(
                                tfLiteDelegateOutputId);
                        results.push_back(tfLiteDelegateOutputData);

                        if (isNumpyOutput)
                        {
                            armnnNumpy::WriteToNumpyFile(m_Params.m_OutputTensorFiles[outputIndex],
                                                         tfLiteDelegateOutputData,
                                                         outputSize,
                                                         dataType,
                                                         shape);
                        }
                        else
                        {
                            for (int i = 0; i < outputSize; ++i)
                            {
                                fprintf(outputTensorFile, "%u ", tfLiteDelegateOutputData[i]);
                            }
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
        }
        CheckInferenceTimeThreshold(duration, m_Params.m_ThresholdTime);
    }

    return results;
}

void TfLiteExecutor::CompareAndPrintResult(std::vector<const void*> otherOutput)
{
    for (unsigned int outputIndex = 0; outputIndex < m_TfLiteInterpreter->outputs().size(); ++outputIndex)
    {
        auto tfLiteDelegateOutputId = m_TfLiteInterpreter->outputs()[outputIndex];
        size_t size                 = m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->bytes;
        double result = 1; // Presume failure.
        switch (m_TfLiteInterpreter->tensor(tfLiteDelegateOutputId)->type)
        {
            case kTfLiteFloat32:
            {
                auto tfLiteDelegateOutputData = m_TfLiteInterpreter->typed_tensor<float>(tfLiteDelegateOutputId);
                result = ComputeByteLevelRMSE(tfLiteDelegateOutputData, otherOutput[outputIndex], size);
                break;
            }
            case kTfLiteInt32:
            {
                auto tfLiteDelegateOutputData = m_TfLiteInterpreter->typed_tensor<int32_t>(tfLiteDelegateOutputId);
                result = ComputeByteLevelRMSE(tfLiteDelegateOutputData, otherOutput[outputIndex], size);
                break;
            }
            case kTfLiteUInt8:
            {
                auto tfLiteDelegateOutputData = m_TfLiteInterpreter->typed_tensor<uint8_t>(tfLiteDelegateOutputId);
                result = ComputeByteLevelRMSE(tfLiteDelegateOutputData, otherOutput[outputIndex], size);
                break;
            }
            case kTfLiteInt8:
            {
                auto tfLiteDelegateOutputData = m_TfLiteInterpreter->typed_tensor<int8_t>(tfLiteDelegateOutputId);
                result = ComputeByteLevelRMSE(tfLiteDelegateOutputData, otherOutput[outputIndex], size);
                break;
            }
            case kTfLiteBool:
            {
                auto tfLiteDelegateOutputData = m_TfLiteInterpreter->typed_tensor<bool>(tfLiteDelegateOutputId);
                result = ComputeByteLevelRMSE(tfLiteDelegateOutputData, otherOutput[outputIndex], size);
                break;
            }
            default:
            {
                LogAndThrow("Unsupported output type");
            }
        }
        std::cout << "Byte level root mean square error: " << result << "\n";
    }
};
