//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ClassicDelegateUtils.hpp>

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

namespace armnnDelegate
{
TfLiteStatus VisitGatherNdOperator(DelegateData& delegateData,
                                 TfLiteContext* tfLiteContext,
                                 TfLiteNode* tfLiteNode,
                                 int nodeIndex,
                                 int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;

    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteIndicesTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (!IsValid(tfLiteContext, tfLiteIndicesTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& indicesTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteIndicesTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    armnn::BackendId setBackend;
    if (!delegateData.m_Network)
    {
        // Check if supported
        bool isSupported = false;
        FORWARD_LAYER_SUPPORT_FUNC("GATHER_ND",
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

    return Connect(layer, tfLiteNode, delegateData);
}
} // namespace armnnDelegate