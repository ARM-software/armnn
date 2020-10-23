//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <armnn_delegate.hpp>

#include "tensorflow/lite/kernels/builtin_op_kernels.h"
#include <tensorflow/lite/interpreter.h>

namespace armnnDelegate
{

TEST_SUITE("ArmnnDelegate")
{

TEST_CASE ("ArmnnDelegate Registered")
{
    std::unique_ptr<tflite::impl::Interpreter> tfLiteInterpreter;
    tfLiteInterpreter.reset(new tflite::impl::Interpreter);

    // Create the network
    tfLiteInterpreter->AddTensors(3);
    tfLiteInterpreter->SetInputs({0});
    tfLiteInterpreter->SetOutputs({2});

    TfLiteQuantizationParams quantizationParams;
    tfLiteInterpreter->SetTensorParametersReadWrite(0, kTfLiteFloat32, "", {3}, quantizationParams);
    tfLiteInterpreter->SetTensorParametersReadWrite(1, kTfLiteFloat32, "", {3}, quantizationParams);
    tfLiteInterpreter->SetTensorParametersReadWrite(2, kTfLiteFloat32, "", {3}, quantizationParams);
    TfLiteRegistration* nodeRegistration = tflite::ops::builtin::Register_ABS();
    void* data = malloc(sizeof(int));

    tfLiteInterpreter->AddNodeWithParameters({0}, {2}, nullptr, 0, data, nodeRegistration);

    // create the Armnn Delegate
    auto delegateOptions = TfLiteArmnnDelegateOptionsDefault();
    auto delegate = TfLiteArmnnDelegateCreate(delegateOptions);
    auto status = tfLiteInterpreter->ModifyGraphWithDelegate(std::move(delegate));
    CHECK(status == kTfLiteOk);
    CHECK(tfLiteInterpreter != nullptr);

}

}

} // namespace armnnDelegate
