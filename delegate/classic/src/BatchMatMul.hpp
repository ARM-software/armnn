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
    TfLiteStatus VisitBatchMatMulOperator(DelegateData& delegateData,
                                          TfLiteContext* tfLiteContext,
                                          TfLiteNode* tfLiteNode,
                                          int nodeIndex,
                                          int32_t operatorCode)
    {
        TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
        TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

        const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
        const TfLiteTensor& kTfLiteLHSInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
        const TfLiteTensor& kTfLiteRHSInputTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];

        if (!IsValid(tfLiteContext, kTfLiteLHSInputTensor, operatorCode, nodeIndex))
        {
            return kTfLiteError;
        }
        if (!IsValid(tfLiteContext, kTfLiteRHSInputTensor, operatorCode, nodeIndex))
        {
            return kTfLiteError;
        }

        if (IsDynamicTensor(kTfLiteLHSInputTensor) || IsDynamicTensor(kTfLiteRHSInputTensor))
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
                    operatorCode, nodeIndex);
            return kTfLiteError;
        }

        const TfLiteTensor& kTfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
        if (IsDynamicTensor(kTfLiteOutputTensor))
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
                    operatorCode, nodeIndex);
            return kTfLiteError;
        }

        const armnn::TensorInfo& armnnLHSInputTensorInfo = GetTensorInfoForTfLiteTensor(kTfLiteLHSInputTensor);
        const armnn::TensorInfo& armnnRHSInputTensorInfo = GetTensorInfoForTfLiteTensor(kTfLiteRHSInputTensor);
        const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(kTfLiteOutputTensor, true);

        armnn::BatchMatMulDescriptor descriptor;
        auto* params = reinterpret_cast<TfLiteBatchMatMulParams *>(tfLiteNode->builtin_data);

        // Tensorflow params are called adjoint, however they are actually just transposes behind the scene. They do
        // not perform ajoint.
        descriptor.m_TransposeX = params->adj_x;
        descriptor.m_TransposeY = params->adj_y;

        // Check if supported
        bool isSupported = false;
        armnn::BackendId setBackend;
        auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
        {
            FORWARD_LAYER_SUPPORT_FUNC("BATCH_MATMUL",
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

        armnn::IConnectableLayer* layer = delegateData.m_Network->AddBatchMatMulLayer(descriptor);
        layer->SetBackendId(setBackend);
        ARMNN_ASSERT(layer != nullptr);

        armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
        outputSlot.SetTensorInfo(outputTensorInfo);

        // try to connect the Constant Inputs if there are any
        if(ProcessInputs(layer,delegateData, tfLiteContext, tfLiteNode) != kTfLiteOk )
        {
            return kTfLiteError;
        }

       return Connect(layer, tfLiteNode, delegateData);
    }
} // namespace armnnDelegate
