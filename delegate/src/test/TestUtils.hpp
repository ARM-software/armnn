//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/interpreter.h>

#include <doctest/doctest.h>

#include <half/half.hpp>

using Half = half_float::half;

namespace armnnDelegate
{

/// Can be used to assign input data from a vector to a model input.
/// Example usage can be found in ResizeTesthelper.hpp
template <typename T>
void FillInput(std::unique_ptr<tflite::Interpreter>& interpreter, int inputIndex, std::vector<T>& inputValues)
{
    auto tfLiteDelegateInputId = interpreter->inputs()[inputIndex];
    auto tfLiteDelageInputData = interpreter->typed_tensor<T>(tfLiteDelegateInputId);
    for (unsigned int i = 0; i < inputValues.size(); ++i)
    {
        tfLiteDelageInputData[i] = inputValues[i];
    }
}

template <>
void FillInput(std::unique_ptr<tflite::Interpreter>& interpreter, int inputIndex, std::vector<Half>& inputValues);

/// Can be used to compare bool data coming from a tflite interpreter
/// Boolean types get converted to a bit representation in a vector. vector.data() returns a void pointer
/// instead of a pointer to bool. Therefore a special function to compare to vector of bool is required
void CompareData(std::vector<bool>& tensor1, bool tensor2[], size_t tensorSize);
void CompareData(bool tensor1[], bool tensor2[], size_t tensorSize);

/// Can be used to compare float data coming from a tflite interpreter with a tolerance of limit_of_float*100
void CompareData(float tensor1[], float tensor2[], size_t tensorSize);

/// Can be used to compare float data coming from a tflite interpreter with a given percentage tolerance
void CompareData(float tensor1[], float tensor2[], size_t tensorSize, float percentTolerance);

/// Can be used to compare int8_t data coming from a tflite interpreter with a tolerance of 1
void CompareData(int8_t tensor1[], int8_t tensor2[], size_t tensorSize);

/// Can be used to compare uint8_t data coming from a tflite interpreter with a tolerance of 1
void CompareData(uint8_t tensor1[], uint8_t tensor2[], size_t tensorSize);

/// Can be used to compare int16_t data coming from a tflite interpreter with a tolerance of 1
void CompareData(int16_t tensor1[], int16_t tensor2[], size_t tensorSize);

/// Can be used to compare int32_t data coming from a tflite interpreter with a tolerance of 1
void CompareData(int32_t tensor1[], int32_t tensor2[], size_t tensorSize);

/// Can be used to compare Half (Float16) data with a tolerance of limit_of_float*100
void CompareData(Half tensor1[], Half tensor2[], size_t tensorSize);

/// Can be used to compare TfLiteFloat16 data coming from a tflite interpreter
void CompareData(TfLiteFloat16 tensor1[], TfLiteFloat16 tensor2[], size_t tensorSize);

/// Can be used to compare Half (Float16) data and TfLiteFloat16 data coming from a tflite interpreter
void CompareData(TfLiteFloat16 tensor1[], Half tensor2[], size_t tensorSize);

/// Can be used to compare the output tensor shape and values
/// from armnnDelegateInterpreter and tfLiteInterpreter.
/// Example usage can be found in ControlTestHelper.hpp
template <typename T>
void CompareOutputData(std::unique_ptr<tflite::Interpreter>& tfLiteInterpreter,
                       std::unique_ptr<tflite::Interpreter>& armnnDelegateInterpreter,
                       std::vector<int32_t>& expectedOutputShape,
                       std::vector<T>& expectedOutputValues,
                       unsigned int outputIndex = 0)
{
    auto tfLiteDelegateOutputId = tfLiteInterpreter->outputs()[outputIndex];
    auto tfLiteDelegateOutputTensor = tfLiteInterpreter->tensor(tfLiteDelegateOutputId);
    auto tfLiteDelegateOutputData = tfLiteInterpreter->typed_tensor<T>(tfLiteDelegateOutputId);
    auto armnnDelegateOutputId = armnnDelegateInterpreter->outputs()[outputIndex];
    auto armnnDelegateOutputTensor = armnnDelegateInterpreter->tensor(armnnDelegateOutputId);
    auto armnnDelegateOutputData = armnnDelegateInterpreter->typed_tensor<T>(armnnDelegateOutputId);

    CHECK(expectedOutputShape.size() == tfLiteDelegateOutputTensor->dims->size);
    CHECK(expectedOutputShape.size() == armnnDelegateOutputTensor->dims->size);

    for (size_t i = 0; i < expectedOutputShape.size(); i++)
    {
        CHECK(expectedOutputShape[i] == armnnDelegateOutputTensor->dims->data[i]);
        CHECK(tfLiteDelegateOutputTensor->dims->data[i] == expectedOutputShape[i]);
        CHECK(tfLiteDelegateOutputTensor->dims->data[i] == armnnDelegateOutputTensor->dims->data[i]);
    }

    armnnDelegate::CompareData(expectedOutputValues.data(), armnnDelegateOutputData    , expectedOutputValues.size());
    armnnDelegate::CompareData(tfLiteDelegateOutputData   , expectedOutputValues.data(), expectedOutputValues.size());
    armnnDelegate::CompareData(tfLiteDelegateOutputData   , armnnDelegateOutputData    , expectedOutputValues.size());
}

} // namespace armnnDelegate
