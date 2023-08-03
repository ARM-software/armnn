//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{
TfLiteStatus VisitBatchMatMulOperator(DelegateData& delegateData,
                                      TfLiteOpaqueContext* tfLiteContext,
                                      TfLiteOpaqueNode* tfLiteNode,
                                      int nodeIndex,
                                      int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    // Gather input indices and use to get input tensor.
    auto numInputs = TfLiteOpaqueNodeNumberOfInputs(tfLiteNode);
    const int* inputTensors;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                nodeIndex);
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* kTfLiteLHSInputTensor =
            TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
    const TfLiteOpaqueTensor* kTfLiteRHSInputTensor =
            TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);

    if (!IsValid(tfLiteContext, kTfLiteLHSInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }
    if (!IsValid(tfLiteContext, kTfLiteRHSInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    // Gather output indices and use to get output tensors.
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

    const TfLiteOpaqueTensor* kTfLiteOutputTensor =
            TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
    if (IsDynamicTensor(kTfLiteOutputTensor))
    {
        TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
                operatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& armnnLHSInputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(kTfLiteLHSInputTensor);
    const armnn::TensorInfo& armnnRHSInputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(kTfLiteRHSInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(kTfLiteOutputTensor, true);

    armnn::BatchMatMulDescriptor descriptor;
    auto* params = reinterpret_cast<TfLiteBatchMatMulParams *>(TfLiteOpaqueNodeGetBuiltinData(tfLiteNode));

    // Tensorflow params are called adjoint, however they are actually just transposes behind the scene. They do
    // not perform ajoint.
    descriptor.m_TransposeX = params->adj_x;
    descriptor.m_TransposeY = params->adj_y;

    // Check if supported
    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("BATCH_MATMUL",
                                           tfLiteContext,
                                           IsBatchMatMulSupported,
                                           delegateData.m_Backends,
                                           isSupported,
                                           setBackend,
                                           armnnLHSInputTensorInfo,
                                           armnnRHSInputTensorInfo,
                                           outputTensorInfo,
                                           descriptor);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    auto layerName = GetName(armnn::LayerType::BatchMatMul, nodeIndex);
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddBatchMatMulLayer(descriptor, layerName.c_str());
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(layer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}

} // namespace armnnOpaqueDelegate
