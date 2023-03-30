//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/Exceptions.hpp>

#include <tensorflow/lite/core/c/c_api.h>
#include <tensorflow/lite/kernels/custom_ops_register.h>
#include <tensorflow/lite/kernels/register.h>

#include <type_traits>

namespace delegateTestInterpreter
{

inline TfLiteTensor* GetInputTensorFromInterpreter(TfLiteInterpreter* interpreter, int index)
{
    TfLiteTensor* inputTensor = TfLiteInterpreterGetInputTensor(interpreter, index);
    if(inputTensor == nullptr)
    {
        throw armnn::Exception("Input tensor was not found at the given index: " + std::to_string(index));
    }
    return inputTensor;
}

inline const TfLiteTensor* GetOutputTensorFromInterpreter(TfLiteInterpreter* interpreter, int index)
{
    const TfLiteTensor* outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, index);
    if(outputTensor == nullptr)
    {
        throw armnn::Exception("Output tensor was not found at the given index: " + std::to_string(index));
    }
    return outputTensor;
}

inline TfLiteModel* CreateTfLiteModel(std::vector<char>& data)
{
    TfLiteModel* tfLiteModel = TfLiteModelCreate(data.data(), data.size());
    if(tfLiteModel == nullptr)
    {
        throw armnn::Exception("An error has occurred when creating the TfLiteModel.");
    }
    return tfLiteModel;
}

inline TfLiteInterpreterOptions* CreateTfLiteInterpreterOptions()
{
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    if(options == nullptr)
    {
        throw armnn::Exception("An error has occurred when creating the TfLiteInterpreterOptions.");
    }
    return options;
}

inline tflite::ops::builtin::BuiltinOpResolver GenerateCustomOpResolver(const std::string& opName)
{
    tflite::ops::builtin::BuiltinOpResolver opResolver;
    if (opName == "MaxPool3D")
    {
        opResolver.AddCustom("MaxPool3D", tflite::ops::custom::Register_MAX_POOL_3D());
    }
    else if (opName == "AveragePool3D")
    {
        opResolver.AddCustom("AveragePool3D", tflite::ops::custom::Register_AVG_POOL_3D());
    }
    else
    {
        throw armnn::Exception("The custom op isn't supported by the DelegateTestInterpreter.");
    }
    return opResolver;
}

template<typename T>
inline TfLiteStatus CopyFromBufferToTensor(TfLiteTensor* tensor, std::vector<T>& values)
{
    // Make sure there is enough bytes allocated to copy into for uint8_t and int16_t case.
    if(tensor->bytes < values.size() * sizeof(T))
    {
        throw armnn::Exception("Tensor has not been allocated to match number of values.");
    }

    // Requires uint8_t and int16_t specific case as the number of bytes is larger than values passed when creating
    // TFLite tensors of these types. Otherwise, use generic TfLiteTensorCopyFromBuffer function.
    TfLiteStatus status = kTfLiteOk;
    if (std::is_same<T, uint8_t>::value)
    {
        for (unsigned int i = 0; i < values.size(); ++i)
        {
            tensor->data.uint8[i] = values[i];
        }
    }
    else if (std::is_same<T, int16_t>::value)
    {
        for (unsigned int i = 0; i < values.size(); ++i)
        {
            tensor->data.i16[i] = values[i];
        }
    }
    else
    {
        status = TfLiteTensorCopyFromBuffer(tensor, values.data(), values.size() * sizeof(T));
    }
    return status;
}

} // anonymous namespace