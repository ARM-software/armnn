//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/utility/IgnoreUnused.hpp>

#include "OpaqueDelegateUtils.hpp"

#include <tensorflow/lite/builtin_ops.h>
#include <tensorflow/lite/c/builtin_op_data.h>
#include <tensorflow/lite/c/common.h>
#include <tensorflow/lite/minimal_logging.h>
#include <numeric>

namespace armnnOpaqueDelegate
{

TfLiteStatus VisitCastOperator(DelegateData& delegateData,
                               TfLiteOpaqueContext* tfLiteContext,
                               TfLiteOpaqueNode* tfLiteNode,
                               int nodeIndex,
                               int32_t operatorCode)
{
    TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));
    int numInputs = 0;
    const int* inputTensors;
    if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    // This layer only has 1 input, so we can directly assign tensor[0] to a new opaque tensor
    const TfLiteOpaqueTensor*
          tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, inputTensors[numInputs-1]);
    if (!IsValid(tfLiteContext, tfLiteInputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    int numOutputs = 0;
    const int* outputTensors;
    if (TfLiteOpaqueNodeOutputs(tfLiteNode, &outputTensors, &numOutputs) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    // This layer only has 1 output, so we can directly assign tensor[0] to a new opaque tensor
    const TfLiteOpaqueTensor*
          tfLiteOutputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext, outputTensors[numOutputs-1]);
    if (!IsValid(tfLiteContext, tfLiteOutputTensor, operatorCode, nodeIndex))
    {
        return kTfLiteError;
    }

    const armnn::TensorInfo& inputTensorInfo  = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
    const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

    bool             isSupported  = false;
    armnn::BackendId setBackend;
    auto             validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported) {
        FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("CAST",
                                   tfLiteContext,
                                   IsCastSupported,
                                   delegateData.m_Backends,
                                   isSupported,
                                   setBackend,
                                   inputTensorInfo,
                                   outInfo);
    };

    // If the m_Network is a nullptr, this signals that a prerequisite TfLite callback is required to clarify the
    // support for the operator
    // If supported, VisitCastOperator will be called again to add the layer to the network as seen further below
    if (!delegateData.m_Network)
    {
        validateFunc(outputTensorInfo, isSupported);
        return isSupported ? kTfLiteOk : kTfLiteError;
    }

    // Add a Cast layer
    armnn::IConnectableLayer* layer = delegateData.m_Network->AddCastLayer();
    layer->SetBackendId(setBackend);
    ARMNN_ASSERT(layer != nullptr);

    armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
    outputSlot.SetTensorInfo(outputTensorInfo);

    // try to connect the Constant Inputs if there are any
    if (ProcessInputs(layer, delegateData, tfLiteContext, tfLiteNode) != kTfLiteOk)
    {
        return kTfLiteError;
    }

    // Connect
    return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
}
}
