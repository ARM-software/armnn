//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DelegateUtils.hpp"
#include <algorithm>
#include <iterator>
#include <string>
#include <vector>

namespace armnnDelegate
{
TfLiteStatus VisitGatherOperator(DelegateData& delegateData,
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

    auto* gatherParameters = reinterpret_cast<TfLiteGatherParams*>(tfLiteNode->builtin_data);
    auto axis = gatherParameters->axis;

    const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& indicesTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteIndicesTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);
    armnn::GatherDescriptor gatherDescriptor;
    gatherDescriptor.m_Axis = axis;

    auto inputDimensions = static_cast<int32_t>(inputTensorInfo.GetNumDimensions());
    auto indicesDimensions = indicesTensorInfo.GetNumDimensions();
    auto outputDimensions = outputTensorInfo.GetNumDimensions();
    if (((axis < -inputDimensions) && (axis < 0)) || ((axis >= inputDimensions) && (axis > 0)))
    {
        TF_LITE_MAYBE_KERNEL_LOG( tfLiteContext,
            "TfLiteArmnnDelegate: Operation has invalid axis: %d. It is out of bounds [-%d, %d))",
            axis, inputDimensions, inputDimensions);
        return kTfLiteError;
    }
    if (outputDimensions != static_cast<unsigned int>(inputDimensions) + indicesDimensions - 1)
    {
        TF_LITE_MAYBE_KERNEL_LOG( tfLiteContext,
            "Operation has invalid output dimensions: %d. Output must be an (%d + %d - 1)-D tensor",
            outputDimensions, inputDimensions, indicesDimensions);
        return kTfLiteError;
    }

    if (!delegateData.m_Network)
    {
        // Check if supported
        bool isSupported = false;
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsGatherSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputTensorInfo,
                                   indicesTensorInfo,
                                   outputTensorInfo,
                                   gatherDescriptor);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* layer = delegateData.m_Network->AddGatherLayer(gatherDescriptor);
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

    Connect(layer, tfLiteNode, delegateData);

    return kTfLiteOk;
}
} // namespace armnnDelegate