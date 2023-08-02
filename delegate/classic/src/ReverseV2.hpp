//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ClassicDelegateUtils.hpp>

#include <armnn/utility/IgnoreUnused.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <tensorflow/lite/kernels/internal/tensor_ctypes.h>

namespace armnnDelegate
{



TfLiteStatus ValidateReverseV2Operator(DelegateData& delegateData,
                                       TfLiteContext* tfLiteContext,
                                       const armnn::TensorInfo& inputInfo0,
                                       const armnn::TensorInfo& inputInfo1,
                                       const armnn::TensorInfo& outputInfo)
{
    bool isSupported = false;
    FORWARD_LAYER_SUPPORT_FUNC("REVERSEV2",
                               tfLiteContext,
                               IsReverseV2Supported,
                               delegateData.m_Backends,
                               isSupported,
                               armnn::BackendId(),
                               inputInfo0,
                               inputInfo1,
                               outputInfo);

    return isSupported ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus VisitReverseV2Operator(DelegateData& delegateData,
                                    TfLiteContext* tfLiteContext,
                                    TfLiteNode* tfLiteNode,
                                    int nodeIndex,
                                    int32_t reverseV2OperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;

    // The first input contains the data that should be reversed
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (IsDynamicTensor(tfLiteInputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            reverseV2OperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // The second input contains an axis tensor.
    const TfLiteTensor& tfLiteAxisTensor = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (IsDynamicTensor(tfLiteAxisTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            reverseV2OperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // Get the output tensor
    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            reverseV2OperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo0 = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& inputTensorInfo1 = GetTensorInfoForTfLiteTensor(tfLiteAxisTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    if (inputTensorInfo0.GetNumDimensions() != outputTensorInfo.GetNumDimensions())
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: input tensor dimension and output tensor dimension differ #%d node #%d: ",
            reverseV2OperatorCode, nodeIndex);
        return kTfLiteError;
    }

    for (unsigned i=0; i < inputTensorInfo0.GetNumDimensions(); i++)
    {
        if (inputTensorInfo0.GetShape()[i] != outputTensorInfo.GetShape()[i])
        {
            TF_LITE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnDelegate: input tensor dimension and output tensor differ #%d node #%d: ",
                reverseV2OperatorCode, nodeIndex);
            return kTfLiteError;
        }
    }

    const auto maxDimension = 4;

    const auto axisTensorNumValues = static_cast<unsigned int>(tfLiteAxisTensor.dims->size);
    if (axisTensorNumValues > maxDimension)
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: The Axis-Input-Tensor of the ReverseV2 operation requires a "
            "dimension of <= %d but a tensor with a dimension of %d was given.                "
            "Operator: #%d node #%d: ",
            maxDimension, axisTensorNumValues, reverseV2OperatorCode, nodeIndex);
        return kTfLiteError;
    }

    // No network pointer indicates that only support for this operator should be checked
    if (!delegateData.m_Network)
    {
        return ValidateReverseV2Operator(delegateData,
                                         tfLiteContext,
                                         inputTensorInfo0,
                                         inputTensorInfo1,
                                         outputTensorInfo);
    }

    auto layerName = GetLayerName(armnn::LayerType::ReverseV2, nodeIndex);
    armnn::IConnectableLayer* reverseV2Layer = delegateData.m_Network->AddReverseV2Layer(layerName.c_str());

    armnn::IOutputSlot& outputSlot = reverseV2Layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // Try to connect the Constant Inputs if there are any
    if (ProcessInputs(reverseV2Layer, delegateData, tfLiteContext, tfLiteNode, nodeIndex) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    ARMNN_ASSERT(reverseV2Layer != nullptr);

    return Connect(reverseV2Layer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate
