//
// Copyright © 2020, 2023 Arm Ltd and Contributors. All rights reserved.
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

constexpr const char* FILE_IDENTIFIER = "TFL3";

/// Can be used to compare bool data coming from a tflite interpreter
/// Boolean types get converted to a bit representation in a vector. vector.data() returns a void pointer
/// instead of a pointer to bool. Therefore a special function to compare to vector of bool is required
void CompareData(std::vector<bool>& tensor1, std::vector<bool>& tensor2, size_t tensorSize);
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

/// Can be used to compare the output tensor shape
/// Example usage can be found in ControlTestHelper.hpp
void CompareOutputShape(const std::vector<int32_t>& tfLiteDelegateShape,
                        const std::vector<int32_t>& armnnDelegateShape,
                        const std::vector<int32_t>& expectedOutputShape);

/// Can be used to compare the output tensor values
/// Example usage can be found in ControlTestHelper.hpp
template <typename T>
void CompareOutputData(std::vector<T>& tfLiteDelegateOutputs,
                       std::vector<T>& armnnDelegateOutputs,
                       std::vector<T>& expectedOutputValues)
{
    armnnDelegate::CompareData(expectedOutputValues.data(),  armnnDelegateOutputs.data(), expectedOutputValues.size());
    armnnDelegate::CompareData(tfLiteDelegateOutputs.data(), expectedOutputValues.data(), expectedOutputValues.size());
    armnnDelegate::CompareData(tfLiteDelegateOutputs.data(), armnnDelegateOutputs.data(), expectedOutputValues.size());
}

} // namespace armnnDelegate
