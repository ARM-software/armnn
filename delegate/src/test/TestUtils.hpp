//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <tensorflow/lite/interpreter.h>

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

} // namespace armnnDelegate
