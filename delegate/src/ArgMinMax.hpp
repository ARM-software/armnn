//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/kernels/internal/tensor_ctypes.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus VisitArgMinMaxOperator(DelegateData& delegateData,
                                    TfLiteContext* tfLiteContext,
                                    TfLiteNode* tfLiteNode,
                                    int nodeIndex,
                                    int32_t argMinMaxOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteInputTensor, argMinMaxOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, argMinMaxOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    // Get const axis value from model and set it to descriptor.
    const TfLiteTensor& tfLiteAxisTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (!IsValid(tfLiteContext, tfLiteAxisTensor, argMinMaxOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    armnn::ArgMinMaxDescriptor desc;
    // Get the axis value from the input tensor
    switch (tfLiteAxisTensor.type)
    {
        case kTfLiteInt32:
        case kTfLiteInt64:
            desc.m_Axis = tflite::GetTensorData<int>(&tfLiteAxisTensor)[0];
            break;
        default:
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Axis value data type is not supported in operator #%d node #%d: ",
                argMinMaxOperatorCode, nodeIndex);
            return kTfLiteError;
    }

    // If output_type is int32 then set Signed32 else Signed64. Default type is Signed64.
    if (argMinMaxOperatorCode == kTfLiteBuiltinArgMax)
    {
        desc.m_Function = armnn::ArgMinMaxFunction::Max;
        auto* argMaxParameters = reinterpret_cast<TfLiteArgMaxParams*>(tfLiteNode->builtin_data);
        if (argMaxParameters->output_type != kTfLiteInt32 && argMaxParameters->output_type != kTfLiteInt64)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: output_type data type is not supported in operator #%d node #%d: ",
                argMinMaxOperatorCode, nodeIndex);
            return kTfLiteError;
        }
    }
    else
    {
        desc.m_Function = armnn::ArgMinMaxFunction::Min;
        auto* argMinParameters = reinterpret_cast<TfLiteArgMinParams*>(tfLiteNode->builtin_data);
        if (argMinParameters->output_type != kTfLiteInt32 && argMinParameters->output_type != kTfLiteInt64)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: output_type data type is not supported in operator #%d node #%d: ",
                    argMinMaxOperatorCode, nodeIndex);
            return kTfLiteError;
        }
    }

    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsArgMinMaxSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputTensorInfo,
                                   outInfo,
                                   desc);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Add an ArgMinMax layer
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddArgMinMaxLayer(desc);
    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // Connect
    return Connect(layer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate
