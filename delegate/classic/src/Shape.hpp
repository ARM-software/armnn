//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <ClassicDelegateUtils.hpp>

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <numeric>

namespace armnnDelegate
{

TfLiteStatus VisitShapeOperator(DelegateData& delegateData,
                               TfLiteContext* tfLiteContext,
                               TfLiteNode* tfLiteNode,
                               int nodeIndex,
                               int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

    const TfLiteTensor* tfLiteTensors = tfLiteContext->tensors;
    const TfLiteTensor& tfLiteInputTensor = tfLiteTensors[tfLiteNode->inputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const TfLiteTensor& tfLiteOutputTensor = tfLiteTensors[tfLiteNode->outputs->data[0]];
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteTensor(tfLiteOutputTensor, true);

    auto* shapeParameters = reinterpret_cast<TfLiteShapeParams*>(tfLiteNode->builtin_data);
    if ( shapeParameters->out_type != kTfLiteInt32 && shapeParameters->out_type != kTfLiteInt64 )
    {
        TF_LITE_MAYBE_KERNEL_LOG(
            tfLiteContext,
            "TfLiteArmnnDelegate: output_type data type is not supported in operator #%d node #%d: ",
            operatorCode, nodeIndex);
        return kTfLiteError;
    }

    bool isSupported = false;
    armnn::BackendId setBackend;
    auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
    {
        FORWARD_LAYER_SUPPORT_FUNC("SHAPE",
                                   tfLiteContext,
                                   IsShapeSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo,
                                   outInfo);
    };

    // If the m_Network is a nullptr, this signals that a prerequisite TfLite callback is required to clarify the
    // support for the operator
    // If supported, VisitShapeOperator will be called again to add the layer to the network as seen further below
    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Add a Shape layer
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddShapeLayer();
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if(ProcessInputs(layer,delegateData, tfLiteContext, tfLiteNode) != kTfLiteOk )
    {
        return kTfLiteError;
    }

    // Connect
    return Connect(layer, tfLiteNode, delegateData);
}

} // namespace armnnDelegate
