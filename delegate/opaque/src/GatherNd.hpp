//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

namespace armnnOpaqueDelegate
{
TfLiteStatus VisitGatherNdOperator(DelegateData& delegateData,
                                   TfLiteOpaqueContext* tfLiteContext,
                                   TfLiteOpaqueNode* tfLiteNode,
                                   int nodeIndex,
                                   int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

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

    const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                     inputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteIndicesTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                       inputTensors[1]);
    if (!IsValid(tfLiteContext, tfLiteIndicesTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteOpaqueTensor* tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                      outputTensors[0]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& indicesTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteIndicesTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    armnn::BackendId setBackend;
    if (!delegateData.m_Network)
    {
        // Check if supported
        bool isSupported = false;
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("GATHER_ND",
                                          tfLiteContext,
                                          IsGatherNdSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          setBackend,
                                          inputTensorInfo,
                                          indicesTensorInfo,
                                          outputTensorInfo);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* layer = delegateData.m_Network->AddGatherNdLayer();
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);
    layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

    auto inputsTensorsProcess = ProcessInputs(layer,
                                              delegateData,
                                              tfLiteContext,
                                              tfLiteNode);
    if (inputsTensorsProcess == kTfLiteError)
    {
        return inputsTensorsProcess;
    }

    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}
} // namespace armnnOpaqueDelegate