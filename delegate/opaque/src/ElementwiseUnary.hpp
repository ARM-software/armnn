//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "OpaqueDelegateUtils.hpp"

namespace armnnOpaqueDelegate
{

std::string GetLayerName(armnn::UnaryOperation unaryOperation)
{
    std::string layerName = "ELEMENTWISE_UNARY";
    switch (unaryOperation)
    {
        case armnn::UnaryOperation::Abs:
            layerName += " ABS";
            break;
        case armnn::UnaryOperation::Ceil:
            layerName += " CEIL";
            break;
        case armnn::UnaryOperation::Exp:
            layerName += " EXP";
            break;
        case armnn::UnaryOperation::Log:
            layerName += " LOG";
            break;
        case armnn::UnaryOperation::LogicalNot:
            layerName += " LOGICALNOT";
            break;
        case armnn::UnaryOperation::Neg:
            layerName += " NEG";
            break;
        case armnn::UnaryOperation::Rsqrt:
            layerName += " RSQRT";
            break;
        case armnn::UnaryOperation::Sin:
            layerName += " SIN";
            break;
        case armnn::UnaryOperation::Sqrt:
            layerName += " SQRT";
            break;
        default:
            layerName += " UNKNOWN";
    }
    return layerName;
}

TfLiteStatus VisitElementwiseUnaryOperator(DelegateData& delegateData,
                                           TfLiteOpaqueContext* tfLiteContext,
                                           TfLiteOpaqueNode* tfLiteNode,
                                           int nodeIndex,
                                           int32_t tfLiteElementWiseUnaryOperatorCode,
                                           armnn::UnaryOperation unaryOperation)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Gather input indices and use to get input tensor.
    int numInputs = 0;
    const int* inputTensors;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }
    // Use input indices to get input tensor.
    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, tfLiteElementWiseUnaryOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Gather output indices and use to get output tensor.
    int numOutputs = 0;
    const int* outputTensors;
    if (TfLiteOpaqueNodeOutputs(tfLiteNode, &outputTensors, &numOutputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather output tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }
    // Use output indices to get output tensor.
    const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, tfLiteElementWiseUnaryOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    armnn::ElementwiseUnaryDescriptor descriptor(unaryOperation);
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported, std::string layerName)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC(layerName.c_str(),
                                          tfLiteContext,
                                          IsElementwiseUnarySupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputTensorInfo,
                                          outputTensorInfo,
                                          descriptor);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported, GetLayerName(unaryOperation));
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* layer = delegateData.m_Network->AddElementwiseUnaryLayer(descriptor);
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if(ProcessInputs(layer, delegateData, tfLiteContext, tfLiteNode) != kTfLiteOk )
    {
        return kTfLiteError;
    }

    // Connect
    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}

} // namespace armnnOpaqueDelegate