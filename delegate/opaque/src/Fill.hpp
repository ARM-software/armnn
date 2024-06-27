//
// Copyright © 2023-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#pragma once

#include <OpaqueDelegateUtils.hpp>

namespace armnnOpaqueDelegate
{

    TfLiteStatus VisitFillOperator(DelegateData& delegateData,
                                   TfLiteOpaqueContext* tfLiteContext,
                                   TfLiteOpaqueNode* tfLiteNode,
                                   int nodeIndex,
                                   int32_t tfLiteFillOperatorCode)
    {
        TF_LITE_ENSURE_STATUS(ValidateNumOutputs(tfLiteContext, tfLiteNode, 1, nodeIndex));

        switch(tfLiteFillOperatorCode)
        {
            case kTfLiteBuiltinFill:
                TF_LITE_ENSURE_STATUS(ValidateNumInputs(tfLiteContext, tfLiteNode, 2, nodeIndex));
                break;
            default:
                return kTfLiteError;
        }

        // Inputs
        int numInputs = 0;
        const int* inputTensors;
        if (TfLiteOpaqueNodeInputs(tfLiteNode, &inputTensors, &numInputs) != kTfLiteOk)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnOpaqueDelegate: Unable to gather input tensor indices from node #%d: ",
                    nodeIndex);
            return kTfLiteError;
        }

        const TfLiteOpaqueTensor* tfLiteInputTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                         inputTensors[0]);
        if (!IsValid(tfLiteContext, tfLiteInputTensor, tfLiteFillOperatorCode, nodeIndex))
        {
            return kTfLiteError;
        }

        const TfLiteOpaqueTensor* tfLiteFillTensor = TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                        inputTensors[1]);

        if(TfLiteOpaqueTensorGetAllocationType(tfLiteFillTensor) != kTfLiteMmapRo)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                tfLiteContext,
                "TfLiteArmnnOpaqueDelegate: FILL tensor must be constant - not supported in operator #%d node #%d: ",
                tfLiteFillOperatorCode, nodeIndex);
            return kTfLiteError;
        }

        if (!IsValid(tfLiteContext, tfLiteFillTensor, tfLiteFillOperatorCode, nodeIndex))
        {
            return kTfLiteError;
        }

        int numOutputs = 0;
        const int* outputTensors;
        if (TfLiteOpaqueNodeOutputs(tfLiteNode, &outputTensors, &numOutputs) != kTfLiteOk)
        {
            TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                    tfLiteContext,
                    "TfLiteArmnnOpaqueDelegate: Unable to gather output tensor indices from node #%d: ",
                    nodeIndex);
            return kTfLiteError;
        }

        const TfLiteOpaqueTensor* tfLiteOutputTensor =  TfLiteOpaqueContextGetOpaqueTensor(tfLiteContext,
                                                                                           outputTensors[0]);
        if (!IsValid(tfLiteContext, tfLiteOutputTensor, tfLiteFillOperatorCode, nodeIndex))
        {
            return kTfLiteError;
        }

        armnn::TensorInfo inputTensorInfo  = GetTensorInfoForTfLiteOpaqueTensor(tfLiteInputTensor);
        const armnn::TensorInfo& outputTensorInfo = GetTensorInfoForTfLiteOpaqueTensor(tfLiteOutputTensor, true);

        armnn::FillDescriptor descriptor;
        switch (TfLiteOpaqueTensorType(tfLiteFillTensor))
        {
            case kTfLiteFloat32:
                descriptor.m_Value = *static_cast<float*>(TfLiteOpaqueTensorData(tfLiteFillTensor));
                break;
            case kTfLiteInt32:
                descriptor.m_Value = *static_cast<int32_t*>(TfLiteOpaqueTensorData(tfLiteFillTensor));
                break;
            default:
                TF_LITE_OPAQUE_MAYBE_KERNEL_LOG(
                        tfLiteContext,
                        "TfLiteArmnnOpaqueDelegate: FILL value data type is not supported in operator #%d node #%d: ",
                        tfLiteFillOperatorCode, nodeIndex);
                return kTfLiteError;
        }

        bool isSupported = false;
        armnn::BackendId setBackend;
        auto validateFunc = [&](const armnn::TensorInfo& outInfo, bool& isSupported)
        {
            FORWARD_LAYER_OPAQUE_SUPPORT_FUNC("FILL",
                                       tfLiteContext,
                                       IsFillSupported,
                                       delegateData.m_Backends,
                                       isSupported,
                                       setBackend,
                                       inputTensorInfo,
                                       outInfo,
                                       descriptor);
        };

        if (!delegateData.m_Network)
        {
            validateFunc(outputTensorInfo, isSupported);
            return isSupported ? kTfLiteOk : kTfLiteError;
        }

        auto layerName = GetName(armnn::LayerType::Fill, nodeIndex);
        armnn::IConnectableLayer* layer = delegateData.m_Network->AddFillLayer(descriptor, layerName.c_str());
        layer->SetBackendId(setBackend);
        ARMNN_ASSERT(layer != nullptr);

        armnn::IOutputSlot& outputSlot = layer->GetOutputSlot(0);
        outputSlot.SetTensorInfo(outputTensorInfo);

        auto inputsTensorsProcess = ProcessInputs(layer,
                                                  delegateData,
                                                  tfLiteContext,
                                                  tfLiteNode,
                                                  nodeIndex);
        if (inputsTensorsProcess == kTfLiteError)
        {
            return inputsTensorsProcess;
        }

        return Connect(layer, tfLiteContext, tfLiteNode, delegateData);
    }

} // namespace armnnDelegate
