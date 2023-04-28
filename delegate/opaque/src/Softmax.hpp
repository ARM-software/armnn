//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{
TfLiteStatus ValidateSoftmaxOperator(DelegateData& delegateData,
                                     TfLiteOpaqueContext* tfLiteContext,
                                     const armnn::TensorInfo& inputInfo,
                                     const armnn::TensorInfo& outputTensorInfo,
                                     const armnn::SoftmaxDescriptor& descriptor)
{
    bool isSupported = false;
    FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("SOFTMAX",
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
                                        TfLiteOpaqueContext* tfLiteContext,
                                        const armnn::TensorInfo& inputInfo,
                                        const armnn::TensorInfo& outputTensorInfo,
                                        const armnn::LogSoftmaxDescriptor& descriptor)
{
    bool isSupported = false;
    FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("LOG_SOFTMAX",
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
                                  TfLiteOpaqueContext* tfLiteContext,
                                  TfLiteOpaqueNode* tfLiteNode,
                                  int nodeIndex,
                                  int32_t tfliteSoftmaxOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Gather input indices and use to get input tensor.
    const int* inputTensors;
    int numInputs;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, tfliteSoftmaxOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Gather output indices and use to get output tensor.
    const int* outputTensors;
    int numOutputs = 0;
    if (TfLiteOpaqueNodeOutputs(tfLiteNode, &outputTensors, &numOutputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather output tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, tfliteSoftmaxOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    if (!delegateData.m_Network)
    {
        switch(tfliteSoftmaxOperatorCode)
        {
            case kTfLiteBuiltinSoftmax:
            {
                armnn::SoftmaxDescriptor descriptor;
                auto* nodeParams = reinterpret_cast<TfLiteSoftmaxParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
                descriptor.m_Beta = nodeParams->beta;
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
    switch(tfliteSoftmaxOperatorCode)
    {
        case kTfLiteBuiltinSoftmax:
        {
            armnn::SoftmaxDescriptor descriptor;
            auto* nodeParameters = reinterpret_cast<TfLiteSoftmaxParams*>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));
            descriptor.m_Beta = nodeParameters->beta;
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
    return Connect(softmaxLayer, tfLiteContext, tfLiteNode, delegateData);
}
} // namespace armnnOpaqueDelegate