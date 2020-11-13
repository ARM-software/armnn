//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <tensorflow/lite/interpreter.h>

#include <doctest/doctest.h>

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

// Can be used to compare the output tensor shape and values
// from armnnDelegateInterpreter and tfLiteInterpreter.
// Example usage can be found in ControlTestHelper.hpp
template <typename T>
void CompareOutputData(std::unique_ptr<tflite::Interpreter>& tfLiteInterpreter,
                       std::unique_ptr<tflite::Interpreter>& armnnDelegateInterpreter,
                       std::vector<int32_t>& expectedOutputShape,
                       std::vector<T>& expectedOutputValues)
{
    auto tfLiteDelegateOutputId = tfLiteInterpreter->outputs()[0];
    auto tfLiteDelegateOutputTensor = tfLiteInterpreter->tensor(tfLiteDelegateOutputId);
    auto tfLiteDelageOutputData = tfLiteInterpreter->typed_tensor<T>(tfLiteDelegateOutputId);
    auto armnnDelegateOutputId = armnnDelegateInterpreter->outputs()[0];
    auto armnnDelegateOutputTensor = armnnDelegateInterpreter->tensor(armnnDelegateOutputId);
    auto armnnDelegateOutputData = armnnDelegateInterpreter->typed_tensor<T>(armnnDelegateOutputId);

    for (size_t i = 0; i < expectedOutputShape.size(); i++)
    {
        CHECK(expectedOutputShape[i] == armnnDelegateOutputTensor->dims->data[i]);
        CHECK(tfLiteDelegateOutputTensor->dims->data[i] == expectedOutputShape[i]);
        CHECK(tfLiteDelegateOutputTensor->dims->data[i] == armnnDelegateOutputTensor->dims->data[i]);
    }

    for (size_t i = 0; i < expectedOutputValues.size(); i++)
    {
        CHECK(expectedOutputValues[i] == armnnDelegateOutputData[i]);
        CHECK(tfLiteDelageOutputData[i] == expectedOutputValues[i]);
        CHECK(tfLiteDelageOutputData[i] == armnnDelegateOutputData[i]);
    }
}

} // namespace armnnDelegate
