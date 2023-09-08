//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{
    TfLiteStatus ValidateBroadcastToOperator(DelegateData& delegateData,
                                             TfLiteOpaqueContext *tfLiteContext,
                                             const armnn::TensorInfo& inputInfo,
                                             const armnn::TensorInfo& outputInfo,
                                             const armnn::BroadcastToDescriptor& descriptor)
    {
        bool isSupported = false;
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("BROADCAST_TO",
                                          tfLiteContext,
                                          IsBroadcastToSupported,
                                          delegateData.m_Backends,
                                          isSupported,
                                          armnn::BackendId(),
                                          inputInfo,
                                          outputInfo,
                                          descriptor);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    TfLiteStatus VisitBroadcastToOperator(DelegateData& delegateData,
                                          TfLiteOpaqueContext* tfLiteContext,
                                          TfLiteOpaqueNode* tfLiteNode,
                                          int nodeIndex,
                                          int32_t broadcastToOperatorCode)
    {
        TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
        TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

        // Gather input tensors
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

        // Gather output tensors
        int numOutputs = 0;
        const int* outputTensors;
        if (TfLiteOpaqueNodeOutputs(tfLiteNode, &outputTensors,
                                    &numOutputs) != kTfLiteOk)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Unable to gather output tensor indices from node #%d: ",
                nodeIndex);
            return kTfLiteError;
        }

        // The input contains the data
        const TfLiteOpaqueTensor* tfLiteInputTensor =
                TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[0]);
        if (IsDynamicTensor(tfLiteInputTensor))
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
                broadcastToOperatorCode, nodeIndex);
            return kTfLiteError;
        }

        // The shape tensor
        const TfLiteOpaqueTensor* tfLiteShapeTensor =
                TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[1]);;
        if (IsDynamicTensor(tfLiteShapeTensor))
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
                broadcastToOperatorCode, nodeIndex);
            return kTfLiteError;
        }

        // The output tensor
        const TfLiteOpaqueTensor* tfLiteOutputTensor =
                TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[0]);
        if (IsDynamicTensor(tfLiteOutputTensor))
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
                broadcastToOperatorCode, nodeIndex);
            return kTfLiteError;
        }

        const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
        const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor,
                                                                                       true);

        auto* shapeData = static_cast<int32_t*>(TfLiteOpaqueTensorData(tfLiteShapeTensor));
        int32_t shapeTensorNum = TfLiteOpaqueTensorDim(tfLiteShapeTensor, 0);

        armnn::BroadcastToDescriptor broadcastToDescriptor;
        broadcastToDescriptor.m_BroadcastToShape = armnn::TensorShape(shapeTensorNum,
                                                                      shapeData);

        // No network pointer indicates that only support for this operator should be checked
        if (!delegateData.m_Network)
        {
            return ValidateBroadcastToOperator(delegateData,
                                               tfLiteContext,
                                               inputTensorInfo,
                                               outputTensorInfo,
                                               broadcastToDescriptor);
        }

        std::string layerName("BroadcastTo");
        armnn::IConnectableLayer* layer = delegateData.m_Network->AddBroadcastToLayer(broadcastToDescriptor,
                                                                                      layerName.c_str());

        if (layer == nullptr)
        {
            return kTfLiteError;
        }

        layer->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

        if (ProcessInputs(layer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
        {
            return kTfLiteError;
        }

        return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
    }

} // namespace armnnOpaqueDelegate