//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ClassicDelegateUtils.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus ValidateSoftmaxOperator(DelegateData& delegateData,
                                     TfLiteContext* tfLiteContext,
                                     const armnn::TensorInfo& inputInfo,
                                     const armnn::TensorInfo& outputTensorInfo,
                                     const armnn::SoftmaxDescriptor& descriptor)
{
    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC("SOFTMAX",
                               tfLiteContext,
                               IsSoftmaxSupported,
                               delegateData.m_Backends,
                               isSupported,
                               armnn::BackendId(),
                               inputInfo,
                               outputTensorInfo,
                               descriptor);
    return isSupported ? kTfLiteOk : kTfLiteError;
}


TfLiteStatus ValidateLogSoftmaxOperator(DelegateData& delegateData,
                                        TfLiteContext* tfLiteContext,
                                        const armnn::TensorInfo& inputInfo,
                                        const armnn::TensorInfo& outputTensorInfo,
                                        const armnn::LogSoftmaxDescriptor& descriptor)
{
    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC("LOG_SOFTMAX",
                               tfLiteContext,
                               IsLogSoftmaxSupported,
                               delegateData.m_Backends,
                               isSupported,
                               armnn::BackendId(),
                               inputInfo,
                               outputTensorInfo,
                               descriptor);
    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus VisitSoftmaxOperator(DelegateData& delegateData,
                                  TfLiteContext* tfLiteContext,
                                  TfLiteNode* tfLiteNode,
                                  int nodeIndex,
                                  int32_t softmaxOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in node #%d: ",
            nodeIndex);
        return kTfLiteError;
    }
    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in node #%d: ",
            nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);


    if (!delegateData.m_Network)
    {
        switch(softmaxOperatorCode)
        {
            case kTfLiteBuiltinSoftmax:
            {
                armnn::SoftmaxDescriptor descriptor;
                auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(tfLiteNode->builtin_data);
                descriptor.m_Beta = params->beta;
                return ValidateSoftmaxOperator(delegateData,
                                               tfLiteContext,
                                               inputTensorInfo,
                                               outputTensorInfo,
                                               descriptor);
            }
            case kTfLiteBuiltinLogSoftmax:
            {
                armnn::LogSoftmaxDescriptor descriptor;
                return ValidateLogSoftmaxOperator(delegateData,
                                                  tfLiteContext,
                                                  inputTensorInfo,
                                                  outputTensorInfo,
                                                  descriptor);
            }
            default:
                return kTfLiteError;
        }
    }

    armnn::IConnectableLayer* softmaxLayer = nullptr;

    switch(softmaxOperatorCode)
    {
        case kTfLiteBuiltinSoftmax:
        {
            armnn::SoftmaxDescriptor descriptor;
            auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(tfLiteNode->builtin_data);
            descriptor.m_Beta = params->beta;
            softmaxLayer = delegateData.m_Network->AddSoftmaxLayer(descriptor);
            break;
        }
        case kTfLiteBuiltinLogSoftmax:
        {
            armnn::LogSoftmaxDescriptor descriptor;
            softmaxLayer = delegateData.m_Network->AddLogSoftmaxLayer(descriptor);
            break;
        }
        default:
            return kTfLiteError;
    }
    ARMNN_ASSERT(softmaxLayer != nullptr);

    armnn::IOutputSlot& outputSlot = softmaxLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if(ProcessInputs(softmaxLayer,delegateData, tfLiteContext, tfLiteNode) != kTfLiteOk )
    {
        return kTfLiteError;
    }

    // Connect
    return Connect(softmaxLayer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate
