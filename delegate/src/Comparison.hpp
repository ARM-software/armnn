//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include "DelegateUtils.hpp"
#include <armnn/utility/IgnoreUnused.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus VisitComparisonOperator(DelegateData& delegateData,
                                     TfLiteContext* tfLiteContext,
                                     TfLiteNode* tfLiteNode,
                                     int nodeIndex,
                                     int32_t tfLiteComparisonOperatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor0 = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (IsDynamicTensor(tfLiteInputTensor0))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            tfLiteComparisonOperatorCode, nodeIndex);

        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteInputTensor1 = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (IsDynamicTensor(tfLiteInputTensor1))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic input tensors are not supported in operator #%d node #%d: ",
            tfLiteComparisonOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (IsDynamicTensor(tfLiteOutputTensor))
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: Dynamic output tensors are not supported in operator #%d node #%d: ",
            tfLiteComparisonOperatorCode, nodeIndex);
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo0 = GetTensorInfoForTfLiteTensor(tfLiteInputTensor0);
    const armnn::TensorInfo& inputTensorInfo1 = GetTensorInfoForTfLiteTensor(tfLiteInputTensor1);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    armnn::ComparisonOperation comparisonOperation = armnn::ComparisonOperation::Equal;
    switch(tfLiteComparisonOperatorCode)
    {
        case kTfLiteBuiltinEqual:
            comparisonOperation = armnn::ComparisonOperation::Equal;
            break;
        case kTfLiteBuiltinGreater:
            comparisonOperation = armnn::ComparisonOperation::Greater;
            break;
        case kTfLiteBuiltinGreaterEqual:
            comparisonOperation = armnn::ComparisonOperation::GreaterOrEqual;
            break;
        case kTfLiteBuiltinLess:
            comparisonOperation = armnn::ComparisonOperation::Less;
            break;
        case kTfLiteBuiltinLessEqual:
            comparisonOperation = armnn::ComparisonOperation::LessOrEqual;
            break;
        case kTfLiteBuiltinNotEqual:
            comparisonOperation = armnn::ComparisonOperation::NotEqual;
            break;
        default:
            return kTfLiteError;
    }

    armnn::ComparisonDescriptor descriptor(comparisonOperation);
    bool isSupported = false;

    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsComparisonSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputTensorInfo0,
                                   inputTensorInfo1,
                                   outputTensorInfo,
                                   descriptor);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* comparisonLayer = delegateData.m_Network->AddComparisonLayer(descriptor);
    ARMNN_ASSERT(comparisonLayer != nullptr);

    armnn::IOutputSlot& outputSlot = comparisonLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    auto reshapeLayer = BroadcastTensor(inputTensorInfo0,
                                        inputTensorInfo1,
                                        comparisonLayer,
                                        tfLiteContext,
                                        tfLiteNode,
                                        delegateData);
    if (!reshapeLayer)
    {
        return kTfLiteError;
    }
    return kTfLiteOk;
}

} // namespace armnnDelegate
