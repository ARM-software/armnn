//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus VisitPadOperator(DelegateData& delegateData,
                              TfLiteContext* tfLiteContext,
                              TfLiteNode* tfLiteNode,
                              int nodeIndex,
                              int32_t tfLitePadOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    switch(tfLitePadOperatorCode)
    {
        case kTfLiteBuiltinMirrorPad:
        case kTfLiteBuiltinPad:
            TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
            break;
        case kTfLiteBuiltinPadv2:
            TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 3, nodeIndex));
            break;
        default:
            return kTfLiteError;
    }

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    const TfLiteTensor& tfLitepaddingTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];

    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            tfLitePadOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            tfLitePadOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& paddingTensorInfo = GetTensorInfoForTfLiteTensor(tfLitepaddingTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    // Get the padding data from the input tensor
    auto* paddingData = tflite::GetTensorData<int32_t>(&tfLitepaddingTensor);

    size_t step = 2;
    armnn::PadDescriptor descriptor;
    for (unsigned int i = 0; i < paddingTensorInfo.GetNumElements() / step; ++i)
    {
        descriptor.m_PadList.emplace_back(paddingData[i * step], paddingData[i * step + 1]);
    }

    if (tfLitePadOperatorCode == kTfLiteBuiltinPad && inputTensorInfo.IsQuantized())
    {
        descriptor.m_PadValue = inputTensorInfo.GetQuantizationOffset();
    }
    else if (tfLitePadOperatorCode == kTfLiteBuiltinPadv2)
    {
        const TfLiteTensor& tfLitepaddingValue = tfLiteTensors[tfLiteNode->inputs->data[2]];
        armnn::TensorInfo paddingValueTensorInfo = GetTensorInfoForTfLiteTensor(tfLitepaddingValue);
        if (paddingValueTensorInfo.GetNumElements() != 1)
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Multiple padding value are not supported in operator #%d node #%d: ",
                tfLitePadOperatorCode, nodeIndex);
            return kTfLiteError;
        }
        // Get the padding value from the input tensor
        switch (tfLitepaddingValue.type)
        {
            case kTfLiteFloat32:
                descriptor.m_PadValue = tflite::GetTensorData<float>(&tfLitepaddingValue)[0];
                break;
            case kTfLiteUInt8:
                descriptor.m_PadValue = tflite::GetTensorData<uint8>(&tfLitepaddingValue)[0];
                break;
            case kTfLiteInt8:
                descriptor.m_PadValue = tflite::GetTensorData<int8>(&tfLitepaddingValue)[0];
                break;
            default:
                TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: Padding value datatype is not supported in operator #%d node #%d: ",
                    tfLitePadOperatorCode, nodeIndex);
                return kTfLiteError;
        }
    }
    else if (tfLitePadOperatorCode == kTfLiteBuiltinMirrorPad)
    {
        TfLiteMirrorPaddingParams* options = reinterpret_cast<TfLiteMirrorPaddingParams*>(tfLiteNode->builtin_data);


        if (options->mode == TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingReflect)
        {
            descriptor.m_PaddingMode = armnn::PaddingMode::Reflect;
        }
        else if (options->mode == TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingSymmetric)
        {
            descriptor.m_PaddingMode = armnn::PaddingMode::Symmetric;
        }
        else
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: PaddingMode must be either REFLECT or SYMMETRIC in operator #%d node #%d: ",
                tfLitePadOperatorCode, nodeIndex);
        }

        // If padding mode is Reflect then both paddings must be no greater than inputShape(i) - 1.
        // If padding mode is Symmetric then both paddings must be no greater than inputShape(i).
        auto inputShape = inputTensorInfo.GetShape();
        auto padList = descriptor.m_PadList;

        const unsigned int isReflect =
                static_cast<unsigned int>(descriptor.m_PaddingMode == armnn::PaddingMode::Reflect);
        for(unsigned int i = 0; i < padList.size(); ++i)
        {
            if(padList.at(i).first > (inputShape[i] - isReflect) ||
               padList.at(i).second > (inputShape[i] - isReflect))
            {
                TF_LITE_MAYBE_KERNEL_LOG(
                        tfLiteContext,
                        "TfLiteArmnnDelegate: Padding values must be less (Reflect) or "
                        "equal (Symmetric) to the dimension size in operator #%d node #%d: ",
                        tfLitePadOperatorCode, nodeIndex);
            }
        }
    }

    if (!delegateData.m_Network)
    {
        bool isSupported = false;
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsPadSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputTensorInfo,
                                   outputTensorInfo,
                                   descriptor);

        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* padLayer = delegateData.m_Network->AddPadLayer(descriptor);
    ARMNN_ASSERT(padLayer != nullptr);

    armnn::IOutputSlot& outputSlot = padLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    return Connect(padLayer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate
