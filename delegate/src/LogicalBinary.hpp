//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>

namespace armnnDelegate
{

TfLiteStatus VisitLogicalBinaryOperator(DelegateData& delegateData,
                                        TfLiteContext* tfLiteContext,
                                        TfLiteNode* tfLiteNode,
                                        int nodeIndex,
                                        int32_t logicalOperatorCode,
                                        armnn::LogicalBinaryOperation binaryOperation)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor0 = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteInputTensor0, logicalOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteInputTensor1 = tfLiteTensors[tfLiteNode->inputs->data[1]];
    if (!IsValid(tfLiteContext, tfLiteInputTensor1, logicalOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, logicalOperatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    armnn::TensorInfo inputTensorInfo0 = GetTensorInfoForTfLiteTensor(tfLiteInputTensor0);
    armnn::TensorInfo inputTensorInfo1 = GetTensorInfoForTfLiteTensor(tfLiteInputTensor1);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor);

    // Setup descriptor and assign operation
    armnn::LogicalBinaryDescriptor desc;
    desc.m_Operation = binaryOperation;

    // Check if supported
    bool isSupported = false;
    auto validateFunc = [&](const armnn::TensorInfo& outputTensorInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC(__func__,
                                   tfLiteContext,
                                   IsLogicalBinarySupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   inputTensorInfo0,
                                   inputTensorInfo1,
                                   outputTensorInfo,
                                   desc);
    };

    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    armnn::IConnectableLayer* logicalBinaryLayer = delegateData.m_Network->AddLogicalBinaryLayer(desc);
    ARMNN_ASSERT(logicalBinaryLayer != nullptr);

    armnn::IOutputSlot& outputSlot = logicalBinaryLayer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    auto inputsTensorsProcess = ProcessInputs(logicalBinaryLayer,
                                              delegateData,
                                              tfLiteContext,
                                              tfLiteNode);
    if (inputsTensorsProcess == kTfLiteError)
    {
        return inputsTensorsProcess;
    }

    // LogicalBinary operators support broadcasting
    auto reshapeLayer = BroadcastTensor(inputTensorInfo0,
                                        inputTensorInfo1,
                                        logicalBinaryLayer,
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
