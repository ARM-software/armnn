//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <armnn/utility/IgnoreUnused.hpp>
#include <DelegateUtils.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <tensorflow/lite/kernels/internal/tensor_ctypes.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <armnn_delegate.hpp>

namespace armnnDelegate
{
    TfLiteStatus ValidateBroadcastToOperator(DelegateData& delegateData,
                                             TfLiteContext* tfLiteContext,
                                             const armnn::TensorInfo& inputInfo,
                                             const armnn::TensorInfo& outputInfo,
                                             const armnn::BroadcastToDescriptor& descriptor)
    {
        bool isSupported = false;
        FORWARD_LAYER_SUPPORT_FUNC("BROADCAST_TO",
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
                                          TfLiteContext* tfLiteContext,
                                          TfLiteNode* tfLiteNode,
                                          int nodeIndex,
                                          int32_t broadcastToOperatorCode)
    {
        TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
        TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

        const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;

        // The input contains the data that should be broadcasted
        const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
        if (IsDynamicTensor(tfLiteInputTensor))
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
                broadcastToOperatorCode, nodeIndex);
            return kTfLiteError;
        }

        // The shape tensor contains the new shape to be applied on the input
        const TfLiteTensor& tfLiteShapeTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
        if (IsDynamicTensor(tfLiteShapeTensor))
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
                broadcastToOperatorCode, nodeIndex);
            return kTfLiteError;
        }

        // The output tensor
        const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
        if (IsDynamicTensor(tfLiteOutputTensor))
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
                broadcastToOperatorCode, nodeIndex);
            return kTfLiteError;
        }

        const armnn::TensorInfo& inputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
        const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

        if (ZeroDimPresent({inputTensorInfo, outputTensorInfo}))
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: Zero dimension tensors are not supported in operator #%d node #%d: ",
                broadcastToOperatorCode, nodeIndex);
            return kTfLiteError;
        }

        auto* shapeData = tflite::GetTensorData<int32_t>(&tfLiteShapeTensor);
        auto shapeTensorNum = tfLiteShapeTensor.dims->data[0];

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

        auto layerName = GetLayerName(armnn::LayerType::BroadcastTo, nodeIndex);
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

        return Connect(layer, tfLiteNode, delegateData);
    }

} // namespace armnnDelegate