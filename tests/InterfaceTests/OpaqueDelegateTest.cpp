//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn_delegate.hpp>

#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/core/model.h>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>

int main()
{
    std::unique_ptr<tflite::FlatBufferModel> model;
    model = tflite::FlatBufferModel::BuildFromFile("./simple_conv2d_1_op.tflite");
    if (!model)
    {
        std::cout << "Failed to load TfLite model from: ./simple_conv2d_1_op.tflite" << std::endl;
        return -1;
    }
    std::unique_ptr<tflite::Interpreter> m_TfLiteInterpreter;
    m_TfLiteInterpreter = std::make_unique<tflite::Interpreter>();
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    if (builder(&m_TfLiteInterpreter) != kTfLiteOk)
    {
        std::cout << "Error loading the model into the TfLiteInterpreter." << std::endl;
        return -1;
    }
    // Use default settings until options have been enabled
    flatbuffers::FlatBufferBuilder flatBufferBuilder;
    tflite::TFLiteSettingsBuilder tfliteSettingsBuilder(flatBufferBuilder);
    flatbuffers::Offset<tflite::TFLiteSettings> tfliteSettings = tfliteSettingsBuilder.Finish();
    flatBufferBuilder.Finish(tfliteSettings);
    const tflite::TFLiteSettings* settings =
        flatbuffers::GetRoot<tflite::TFLiteSettings>(flatBufferBuilder.GetBufferPointer());

    std::unique_ptr<tflite::delegates::DelegatePluginInterface> delegatePlugIn =
        tflite::delegates::DelegatePluginRegistry::CreateByName("armnn_delegate", *settings);

    // Create Armnn Opaque Delegate from Armnn Delegate Plugin
    tflite::delegates::TfLiteDelegatePtr armnnDelegate = delegatePlugIn->Create();

    // Add Delegate to the builder
    builder.AddDelegate(armnnDelegate.get());
    if (builder(&m_TfLiteInterpreter) != kTfLiteOk)
    {
        std::cout << "Unable to add the Arm NN delegate to the TfLite runtime." << std::endl;
        return -1;
    }

    if (m_TfLiteInterpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cout << "Failed to allocate tensors in the TfLiteInterpreter." << std::endl;
        return -1;
    }

    // Really should populate the tensors here, but it'll work without it.

    int status = m_TfLiteInterpreter->Invoke();
    if (status != kTfLiteOk)
    {
        std::cout << "Inference failed." << std::endl;
        return -1;
    }
}
