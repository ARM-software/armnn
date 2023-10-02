//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn_delegate.hpp>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/core/c/c_api.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

#include <string>

int main()
{
    std::unique_ptr<tflite::FlatBufferModel> model;
    model = tflite::FlatBufferModel::BuildFromFile("./simple_conv2d_1_op.tflite");
    if (!model)
    {
        std::cout << "Failed to load TfLite model from: ./simple_conv2d_1_op.tflite" << std::endl;
        return -1;
    }
    std::unique_ptr<tflite::Interpreter> tfLiteInterpreter;
    tfLiteInterpreter = std::make_unique<tflite::Interpreter>();
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    if (builder(&tfLiteInterpreter) != kTfLiteOk)
    {
        std::cout << "Error loading the model into the TfLiteInterpreter." << std::endl;
        return -1;
    }

    // Create the Armnn Delegate
    // Populate a DelegateOptions from the ExecuteNetworkParams.
    armnnDelegate::DelegateOptions delegateOptions(armnn::Compute::CpuRef);
    std::unique_ptr<TfLiteDelegate, decltype(&armnnDelegate::TfLiteArmnnDelegateDelete)> theArmnnDelegate(
        armnnDelegate::TfLiteArmnnDelegateCreate(delegateOptions), armnnDelegate::TfLiteArmnnDelegateDelete);
    // Register armnn_delegate to TfLiteInterpreter
    auto result = tfLiteInterpreter->ModifyGraphWithDelegate(std::move(theArmnnDelegate));
    if (result != kTfLiteOk)
    {
        std::cout << "Could not register ArmNN TfLite Delegate to TfLiteInterpreter." << std::endl;
        return -1;
    }
    if (tfLiteInterpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cout << "Failed to allocate tensors in the TfLiteInterpreter." << std::endl;
        return -1;
    }

    // Really should populate the tensors here, but it'll work without it.

    int status = tfLiteInterpreter->Invoke();
    if (status != kTfLiteOk)
    {
        std::cout << "Inference failed." << std::endl;
        return -1;
    }
}
